"""
Script Json to Lerobot (multiprocessing version).

This is functionally identical to ``convert_unitree_json_to_lerobot.py`` but the
conversion is parallelized across processes. The single-process version is
dominated by ``save_episode``, which runs the (CPU-bound) SVT-AV1 video encoding
one episode at a time in the main process.

To use all cores for encoding, this version *shards* the episodes across worker
processes. Each worker builds a complete, standalone ``LeRobotDataset`` for its
contiguous slice of episodes into a temporary directory -- so both the JSON/image
loading AND the video encoding run fully in parallel. The per-shard datasets are
then merged into the final dataset (in shard order, which preserves the original
episode order) using ``lerobot.datasets.aggregate.aggregate_datasets``.

# --raw-dir      Corresponds to the directory of your JSON dataset
# --repo-id      Your unique repo ID on Hugging Face Hub
# --robot_type   The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --push_to_hub  Whether or not to upload the dataset to Hugging Face Hub (true or false)
# --num_workers  Number of worker processes (shards) used to convert episodes (defaults to 1))
# --verbose      Show lerobot's per-episode video-encoding / aggregation logging (off by default)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot_multi.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub
"""

import os
import cv2
import tqdm
import tyro
import json
import glob
import logging
import signal
import dataclasses
import shutil
import contextlib
import concurrent.futures
import multiprocessing as mp
import threading
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

    data_files_size_in_mb: int | None = None
    video_files_size_in_mb: int | None = 500
    chunk_size: int | None = None  # Maximum number of files per chunk


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _set_log_verbosity(verbose: bool) -> None:
    """Quiet lerobot's INFO-level chatter (per-episode video-encoding logs, aggregation
    progress, etc.) unless ``verbose``.

    Must be called in the main process *and* in each spawn worker, since spawned workers
    don't inherit the main process's logging configuration. We raise both the root logger
    (lerobot uses bare ``logging.info(...)`` in some modules) and the ``lerobot`` logger
    (used via module loggers) so the record is dropped before reaching any handler.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    logging.getLogger("lerobot").setLevel(level)

    # HuggingFace ``datasets`` renders "Map: ...%" progress bars during parquet writes;
    # silence those too unless verbose.
    try:
        import datasets

        datasets.enable_progress_bars() if verbose else datasets.disable_progress_bars()
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_native_output(suppress: bool):
    """Temporarily redirect the C-level stdout/stderr fds to /dev/null.

    The SVT-AV1 encoder prints its banner/config (``Svt[info]: ...``) straight to the
    process's stdout/stderr from C, so Python logging levels and ``contextlib.redirect_*``
    (which only swap ``sys.stdout``/``sys.stderr``) don't affect it -- only duplicating the
    underlying file descriptors over /dev/null does. No-op when ``suppress`` is False.

    Worker errors are unaffected: exceptions propagate back through the future to the main
    process, which re-raises them there (with stderr intact).
    """
    if not suppress:
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


class JsonDataset:
    def __init__(
        self,
        data_dirs: Path,
        robot_type: str,
        episode_indices: list[int] | None = None,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
            robot_type: Robot type key into ROBOT_CONFIGS
            episode_indices: Optional subset of (global) episode indices to cache/load. When
                provided, only those episodes' JSON is read into memory -- this lets a worker
                shard avoid caching the whole dataset. The global indexing (``self.episode_paths``)
                is preserved so ``get_item(global_index)`` keeps working.
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"
        self.show_progress = show_progress

        # Initialize paths and cache
        self._init_paths()
        self._init_cache(episode_indices)
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []

        for task_path in glob.glob(os.path.join(self.data_dirs, "*")):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path, "*"))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self, episode_indices: list[int] | None = None) -> dict:
        """Initialize the JSON data cache.

        When ``episode_indices`` is given, only those episodes are cached (keyed by their
        global index); otherwise every episode is cached.
        """

        if episode_indices is None:
            episode_indices = list(range(len(self.episode_paths)))

        self.episodes_data_cached = {}
        for idx in tqdm.tqdm(episode_indices, desc="Loading Cache Json", disable=not self.show_progress):
            json_path = os.path.join(self.episode_paths[idx], self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached[idx] = json.load(jsonf)

        return self.episodes_data_cached

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_image_shapes(self, index: int) -> dict[str, tuple[int, int, int]]:
        """Return the ``(height, width, channel)`` shape of each camera's images.

        Reads a single frame per camera (rather than the whole episode) so the dataset
        features can be declared from the actual data instead of a hardcoded shape.
        """
        episode_path = self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        shapes: dict[str, tuple[int, int, int]] = {}
        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None or image_key in shapes:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                shapes[image_key] = (image.shape[0], image.shape[1], image.shape[2])
                break

        return shapes

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    image_shapes: dict[str, tuple[int, int, int]],
    root: Path | None = None,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        }

    for cam in cameras:
        if cam not in image_shapes:
            raise KeyError(
                f"No image shape found for camera '{cam}'. Available cameras: {sorted(image_shapes)}"
            )
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shapes[cam],
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    # Clean the target location (explicit root for shards, default home otherwise).
    target = Path(root) if root is not None else (HF_LEROBOT_HOME / repo_id)
    if target.exists():
        shutil.rmtree(target)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        root=root,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    json_dataset: JsonDataset,
    episode_indices: list[int],
    progress_queue: "mp.Queue | None" = None,
) -> LeRobotDataset:
    """Populate ``dataset`` with the given (global) episode indices, in order.

    When ``progress_queue`` is given, one item is pushed after each finished episode so
    the main process can render a single global progress bar across all shards (instead
    of one per-worker bar).
    """
    for index in episode_indices:
        episode = json_dataset.get_item(index)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()
        if progress_queue is not None:
            progress_queue.put(1)

    return dataset


def _split_indices(num_episodes: int, num_shards: int) -> list[list[int]]:
    """Split ``range(num_episodes)`` into ``num_shards`` contiguous, near-equal chunks.

    Contiguous chunks keep shards in the original episode order, so aggregating the
    shards in shard order reproduces the single-process episode ordering.
    """
    num_shards = max(1, min(num_shards, num_episodes))
    base, extra = divmod(num_episodes, num_shards)
    shards = []
    start = 0
    for s in range(num_shards):
        size = base + (1 if s < extra else 0)
        if size == 0:
            continue
        shards.append(list(range(start, start + size)))
        start += size
    return shards


def _build_shard(
    shard_id: int,
    episode_indices: list[int],
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"],
    shard_root: str,
    dataset_config: DatasetConfig,
    progress_queue: "mp.Queue | None" = None,
    verbose: bool = False,
) -> tuple[str, str]:
    """Worker: build one standalone single-shard LeRobotDataset and return its (repo_id, root).

    Runs in a separate process. Each worker reads only the JSON for its assigned episodes,
    encodes its own videos (so encoding is parallel across shards), then finalizes the shard.
    Per-worker progress bars are suppressed; episode completion is reported to the main
    process via ``progress_queue`` so a single global bar can be rendered.
    """
    # Spawn workers don't inherit the main process's logging config, so set it here too.
    _set_log_verbosity(verbose)

    shard_repo_id = f"{repo_id}_shard_{shard_id:04d}"
    json_dataset = JsonDataset(raw_dir, robot_type, episode_indices=episode_indices, show_progress=False)

    # Declare image features from the actual data rather than a hardcoded shape.
    image_shapes = json_dataset.get_image_shapes(episode_indices[0])

    dataset = create_empty_dataset(
        shard_repo_id,
        robot_type=robot_type,
        mode=mode,
        image_shapes=image_shapes,
        root=Path(shard_root),
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    # Hide the SVT-AV1 encoder's native stdout/stderr banner unless --verbose. (The global
    # progress bar lives in the main process, so muting this worker's fds doesn't touch it.)
    with _suppress_native_output(suppress=not verbose):
        populate_dataset(dataset, json_dataset, episode_indices, progress_queue=progress_queue)

        # Flush async image writes, encode any pending videos, and write parquet/meta footers.
        dataset.finalize()
    return shard_repo_id, shard_root


def _isolate_process_group() -> None:
    """Worker initializer: put each worker in its own process group.

    This lets the main process tear a worker down together with everything it
    spawns (its per-shard video-encoding pool and the ffmpeg children of that
    pool) via a single ``os.killpg``. It also stops a terminal Ctrl+C -- which
    the OS delivers to the whole foreground process group -- from racing the
    controlled shutdown below: with workers in their own groups, SIGINT reaches
    only the main process, which then drives teardown explicitly.
    """
    os.setpgrp()


def _terminate_executor(executor: concurrent.futures.ProcessPoolExecutor) -> None:
    """Forcefully kill all worker processes and their descendants.

    Each worker is its own process-group leader (see ``_isolate_process_group``),
    so signalling the group also reaches the worker's encoding subprocesses and
    their ffmpeg children. We send SIGTERM first for an orderly exit, then
    SIGKILL any stragglers. Safe to call more than once / after a clean shutdown:
    signals to already-dead processes are ignored.
    """
    own_group = os.getpgrp()

    targets: list[tuple[mp.Process, int | None]] = []
    for proc in list((getattr(executor, "_processes", None) or {}).values()):
        if proc.pid is None:
            continue
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            continue
        # If the worker never isolated itself, its group is ours -- killing that
        # group would take the main process down too, so signal just the worker.
        targets.append((proc, pgid if pgid != own_group else None))

    def _signal(proc: mp.Process, pgid: int | None, sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                os.kill(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    for proc, pgid in targets:
        _signal(proc, pgid, signal.SIGTERM)
    for proc, pgid in targets:
        proc.join(timeout=5)
    for proc, pgid in targets:
        if proc.is_alive():
            _signal(proc, pgid, signal.SIGKILL)

    executor.shutdown(wait=False)


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    num_workers: int | None = 1,
    max_num_episodes: int | None = None,
    verbose: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # Suppress lerobot's per-episode video-encoding / aggregation logging unless --verbose.
    _set_log_verbosity(verbose)

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    # Count episodes without caching all JSON (pass an empty index set to skip the cache).
    num_episodes = len(JsonDataset(raw_dir, robot_type, episode_indices=[]))
    if num_episodes == 0:
        raise ValueError(f"No episodes found under {raw_dir}")
    
    if max_num_episodes is not None:
        num_episodes = min(num_episodes, max_num_episodes)

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    shards = _split_indices(num_episodes, num_workers)
    num_shards = len(shards)

    print(f"==> Converting {num_episodes} episodes across {num_shards} worker shard(s)")

    # Per-shard datasets are written under a temp area inside the HF home so the final
    # aggregation can read them via LeRobotDatasetMetadata(repo_id, root=...).
    shards_root = HF_LEROBOT_HOME / f"{repo_id}_shards_tmp"
    if shards_root.exists():
        shutil.rmtree(shards_root)
    shards_root.mkdir(parents=True, exist_ok=True)

    # Use a non-daemonic process pool (concurrent.futures) because each shard's
    # save_episode() spawns its OWN ProcessPoolExecutor for per-camera encoding;
    # daemonic multiprocessing.Pool workers are not allowed to have children.
    ctx = mp.get_context("spawn")
    built: list[tuple[int, str, str]] = []

    # Single global progress bar: workers push one item per finished episode onto this
    # (manager-backed) queue, and a drain thread advances one bar in the main process.
    manager = ctx.Manager()
    progress_queue = manager.Queue()
    pbar = tqdm.tqdm(total=num_episodes, desc="Converting episodes", unit="ep")

    def _drain_progress() -> None:
        while True:
            item = progress_queue.get()
            if item is None:  # sentinel: stop draining
                break
            pbar.update(item)

    drain_thread = threading.Thread(target=_drain_progress, daemon=True)
    drain_thread.start()

    # Each worker is isolated into its own process group so we can tear it (and its
    # nested encoding pool / ffmpeg children) down cleanly on Ctrl+C or any error.
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_shards, mp_context=ctx, initializer=_isolate_process_group
    )
    try:
        future_to_shard = {
            executor.submit(
                _build_shard,
                shard_id,
                indices,
                raw_dir,
                repo_id,
                robot_type,
                mode,
                str(shards_root / f"shard_{shard_id:04d}"),
                dataset_config,
                progress_queue,
                verbose,
            ): shard_id
            for shard_id, indices in enumerate(shards)
        }
        for future in concurrent.futures.as_completed(future_to_shard):
            shard_id = future_to_shard[future]
            shard_repo_id, shard_root = future.result()
            built.append((shard_id, shard_repo_id, shard_root))

        # All shards finished cleanly -> wait for the workers to exit, then proceed.
        executor.shutdown(wait=True)

        # All progress items are enqueued; stop the drain thread.
        progress_queue.put(None)
        drain_thread.join()
        pbar.close()

        # Aggregate in shard order to preserve the original episode ordering.
        built.sort(key=lambda x: x[0])
        shard_repo_ids = [b[1] for b in built]
        shard_roots = [Path(b[2]) for b in built]

        print(f"==> Aggregating {num_shards} shard(s) into {repo_id}")
        with _suppress_native_output(suppress=not verbose):
            aggregate_datasets(
                repo_ids=shard_repo_ids,
                aggr_repo_id=repo_id,
                roots=shard_roots,
                aggr_root=HF_LEROBOT_HOME / repo_id,
                data_files_size_in_mb=dataset_config.data_files_size_in_mb,
                video_files_size_in_mb=dataset_config.video_files_size_in_mb,
                chunk_size=dataset_config.chunk_size,
            )
    except KeyboardInterrupt:
        print("\n==> Interrupted -- terminating worker processes and their encoders...")
        _terminate_executor(executor)
        raise
    except BaseException:
        # Any other failure (a worker crashed, aggregation errored, ...): kill the
        # remaining workers and their descendants before unwinding.
        _terminate_executor(executor)
        raise
    finally:
        # Safety net: ensure the pool is not left waiting on workers on any path.
        executor.shutdown(wait=False)

        # Tear down the progress bar / drain thread / manager (also covers the error path,
        # where the sentinel above may never have been sent).
        if drain_thread.is_alive():
            progress_queue.put(None)
            drain_thread.join()
        pbar.close()
        manager.shutdown()

        # Always clean up the temporary per-shard datasets.
        if shards_root.exists():
            shutil.rmtree(shards_root)

    if push_to_hub:
        dataset = LeRobotDataset(repo_id=repo_id, root=HF_LEROBOT_HOME / repo_id)
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
