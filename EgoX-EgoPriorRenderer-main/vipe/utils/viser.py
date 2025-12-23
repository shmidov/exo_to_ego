# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import socket
import time
from typing import Tuple, List, Optional

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import viser.transforms as tf

from matplotlib import cm
from PIL import Image
from rich.logging import RichHandler
import pandas as pd
from scipy.spatial.transform import Rotation as R

from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def compute_robust_mean_tensors(stacked_depth: torch.Tensor, stacked_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU 벡터화된 버전: 픽셀별 NaN-aware 평균을 병렬로 계산.
    """
    logger.info("Computing mean background using nanmean with GPU vectorization...")
    
    # NaN을 무시하고 평균 계산
    robust_depth = torch.nanmean(stacked_depth, dim=0)
    robust_rgb = torch.nanmean(stacked_rgb, dim=0)
    
    # NaN 값들을 0으로 대체
    robust_depth = torch.nan_to_num(robust_depth, nan=0.0)
    robust_rgb = torch.nan_to_num(robust_rgb, nan=0.0)
    
    return robust_depth, robust_rgb


@dataclass
class GlobalContext:
    artifacts: list[ArtifactPath]
    use_mean_bg: bool = False
    mean_depth: torch.Tensor = None
    mean_rgb: torch.Tensor = None
    ego_manual: bool = False  # Manual ego camera control mode
    
    def __post_init__(self):
        # dataclass에서 None으로 초기화된 tensor 필드들 처리
        if self.mean_depth is None:
            self.mean_depth = None
        if self.mean_rgb is None:
            self.mean_rgb = None


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None
    dynamic_pcd_handle: viser.PointCloudHandle | None = None

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value
        # Dynamic point cloud는 추가적으로 show_dynamic_objects 설정도 고려
        if self.dynamic_pcd_handle is not None:
            # ClientClosures instance에 접근하기 위해 여기서는 단순히 value만 사용
            # GUI 상태는 GUI 컨트롤의 on_update에서 처리
            self.dynamic_pcd_handle.visible = value


def compute_mean_background_depth(artifact_path: ArtifactPath, 
                                  spatial_subsample: int = 2, max_frames: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    모든 프레임에서 background depth와 RGB의 nanmean을 미리 계산.
    """
    logger.info("Computing mean background depth using nanmean...")
    
    # GPU device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def none_it(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    def none_it_mask(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    valid_depth_frames = []
    valid_rgb_frames = []
    frame_count = 0

    for frame_idx, ((_, rgb), (_, depth), (_, instance_mask)) in enumerate(
        zip(
            read_rgb_artifacts(artifact_path.rgb_path),
            none_it(read_depth_artifacts(artifact_path.depth_path)),
            none_it_mask(read_instance_artifacts(artifact_path.mask_path)),
        )
    ):
        if depth is None:
            continue
        
        if max_frames and frame_count >= max_frames:
            break

        # GPU로 이동
        rgb = rgb.to(device)
        depth = depth.to(device)
        if instance_mask is not None:
            instance_mask = instance_mask.to(device)

        # Background만 유지 (instance_id == 0)
        if instance_mask is not None:
            static_mask = (instance_mask == 0)
        else:
            static_mask = torch.ones_like(depth, dtype=torch.bool, device=device)
        
        # 유효한 depth 마스크
        depth_mask = reliable_depth_mask_range(depth)
        final_mask = static_mask & depth_mask

        # 무효한 픽셀을 NaN으로 설정
        masked_depth = depth.clone().float()
        masked_depth[~final_mask] = float('nan')
        
        masked_rgb = rgb.clone().float()
        masked_rgb[~final_mask.unsqueeze(-1).expand_as(rgb)] = float('nan')

        # Spatial subsampling 적용
        if spatial_subsample > 1:
            masked_depth = masked_depth[::spatial_subsample, ::spatial_subsample]
            masked_rgb = masked_rgb[::spatial_subsample, ::spatial_subsample]

        valid_depth_frames.append(masked_depth)
        valid_rgb_frames.append(masked_rgb)
        
        frame_count += 1
        if frame_idx % 10 == 0:
            logger.info(f"Processed frame {frame_idx} for mean background computation.")

    if not valid_depth_frames:
        logger.warning("No valid background frames found.")
        return None, None

    logger.info(f"Computing nanmean from {len(valid_depth_frames)} frames...")
    
    stacked_depth = torch.stack(valid_depth_frames, dim=0)  # [N_frames, H, W]
    stacked_rgb = torch.stack(valid_rgb_frames, dim=0)      # [N_frames, H, W, 3]
    
    # Nanmean 계산
    mean_depth, mean_rgb = compute_robust_mean_tensors(stacked_depth, stacked_rgb)
    
    logger.info(f"Mean background depth computed: {mean_depth.shape}, device: {mean_depth.device}")
    return mean_depth, mean_rgb


def build_dynamic_points_for_frame(artifact_path: ArtifactPath, frame_idx: int, c2w: np.ndarray, 
                                   rgb: torch.Tensor, depth: torch.Tensor, instance_mask: torch.Tensor,
                                   camera_type: CameraType, intr: torch.Tensor, spatial_subsample: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    특정 프레임에서 dynamic objects (instance_id != 0)의 point cloud를 생성.
    
    Args:
        artifact_path: ViPE artifact path
        frame_idx: Frame index
        c2w: Camera-to-world transformation matrix
        rgb: RGB image tensor
        depth: Depth tensor
        instance_mask: Instance segmentation mask
        camera_type: Camera type
        intr: Camera intrinsics
        spatial_subsample: Spatial subsampling factor
        
    Returns:
        dynamic_points: Nx3 array of 3D points (camera coordinates)
        dynamic_colors: Nx3 array of RGB colors (0-255)
    """
    if depth is None or instance_mask is None:
        return np.empty((0, 3)), np.empty((0, 3))

    frame_height, frame_width = rgb.shape[:2]

    # RGB 처리
    sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
    sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

    # Ray 생성
    camera_model = camera_type.build_camera_model(intr)
    disp_v, disp_u = torch.meshgrid(
        torch.arange(frame_height).float()[::spatial_subsample],
        torch.arange(frame_width).float()[::spatial_subsample],
        indexing="ij",
    )
    if camera_type == CameraType.PANORAMA:
        disp_v = disp_v / (frame_height - 1)
        disp_u = disp_u / (frame_width - 1)
    disp = torch.ones_like(disp_v)
    pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
    rays = pts[..., :3].numpy()
    if camera_type != CameraType.PANORAMA:
        rays /= rays[..., 2:3]

    # Point cloud 생성 (카메라 좌표계)
    pcd_camera = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
    
    # Depth mask 적용
    depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()

    # Dynamic objects만 유지 (instance_id != 0)
    instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
    dynamic_mask = (instance_mask_np != 0)
    dynamic_mask_sub = dynamic_mask[::spatial_subsample, ::spatial_subsample]
    final_mask = depth_mask & dynamic_mask_sub

    # Flatten and filter
    pcd_flat = pcd_camera.reshape(-1, 3)
    rgb_flat = sampled_rgb.reshape(-1, 3)
    mask_flat = final_mask.reshape(-1)

    valid_points = pcd_flat[mask_flat]
    valid_colors = rgb_flat[mask_flat]

    logger.info(f"Frame {frame_idx}: Generated {len(valid_points)} dynamic points")
    
    return valid_points, valid_colors


class ClientClosures:
    """
    All class methods automatically capture 'self', ensuring proper locals.
    """

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        # Don't await to not block the rest of the coroutine.
        self.task = asyncio.create_task(_run())

        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0

        # added
        self.gui_playing: viser.GuiCheckboxHandle | None = None
        self.gui_show_all_frames: viser.GuiCheckboxHandle | None = None
        self.ego_frustum_handles: list[viser.CameraFrustumHandle] = []
        self.ego_trajectory_handles: list = []  # 점선을 위한 여러 spline handles
        self.semidense_pcd_handle: viser.PointCloudHandle | None = None
        
        # Manual ego camera control
        self.manual_ego_frame: viser.FrameHandle | None = None
        self.manual_ego_frustum: viser.CameraFrustumHandle | None = None
        self.manual_ego_transform_handle: viser.TransformControlsHandle | None = None
        self.gui_manual_extrinsic_text: viser.GuiMarkdownHandle | None = None

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        all_artifacts = self.global_context().artifacts

        with self.client.gui.add_folder("Sample"):
            self.gui_id = self.client.gui.add_slider(
                "Artifact ID", min=0, max=len(all_artifacts) - 1, step=1, initial_value=0
            )
            gui_id_changer = self.client.gui.add_button_group(label="ID +/-", options=["Prev", "Next"])

            @gui_id_changer.on_click
            async def _(_) -> None:
                if gui_id_changer.value == "Prev":
                    self.gui_id.value = (self.gui_id.value - 1) % len(all_artifacts)
                else:
                    self.gui_id.value = (self.gui_id.value + 1) % len(all_artifacts)

            self.gui_name = self.client.gui.add_text("Artifact Name", "")
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_id.on_update(self.on_sample_update)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=0.1, step=0.001, initial_value=0.001
            )

            # Update point cloud size
            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value
                    if frame_node.dynamic_pcd_handle is not None:
                        frame_node.dynamic_pcd_handle.point_size = self.gui_point_size.value * 1.2

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox(
                "Colorful Frustum",
                initial_value=False,
            )

            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            # 동적 객체 필터링 컨트롤 추가
            self.gui_filter_dynamic_objects = self.client.gui.add_checkbox(
                "Filter Dynamic Objects",
                initial_value=True,
                hint="Remove dynamic objects (person, car, etc.) from point cloud"
            )

            @self.gui_filter_dynamic_objects.on_update
            async def _(_) -> None:
                await self.on_sample_update(None)

            # Dynamic objects 표시 컨트롤 추가
            self.gui_show_dynamic_objects = self.client.gui.add_checkbox(
                "Show Dynamic Objects",
                initial_value=True,
                hint="Show dynamic objects (people, cars, etc.) as separate point cloud"
            )

            @self.gui_show_dynamic_objects.on_update
            async def _(_) -> None:
                # Dynamic point clouds의 visibility 토글
                for frame_node in self.scene_frame_handles:
                    if frame_node.dynamic_pcd_handle is not None:
                        frame_node.dynamic_pcd_handle.visible = self.gui_show_dynamic_objects.value and frame_node.visible

            # Semi-dense point cloud 표시 컨트롤 추가
            self.gui_show_semidense_pcd = self.client.gui.add_checkbox(
                "Show Semi-dense Points",
                initial_value=True,
                hint="Show Ego4D semi-dense point cloud (pseudo GT)"
            )

            @self.gui_show_semidense_pcd.on_update
            async def _(_) -> None:
                # Semi-dense point cloud visibility 토글
                if hasattr(self, 'semidense_pcd_handle') and self.semidense_pcd_handle is not None:
                    self.semidense_pcd_handle.visible = self.gui_show_semidense_pcd.value

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            gui_snapshot = self.client.gui.add_button(
                "Snapshot",
                hint="Take a snapshot of the current scene",
            )

            # Async get_render does not work at the moment, we will put into thread loop.
            @gui_snapshot.on_click
            def _(_) -> None:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                file_name = f"{current_artifact.base_path.name}_{current_artifact.artifact_name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

        # Manual Ego Camera Control
        if self.global_context().ego_manual:
            with self.client.gui.add_folder("Manual Ego Camera"):
                gui_show_manual_ego = self.client.gui.add_checkbox(
                    "Show Manual Camera",
                    initial_value=True,
                    hint="Show/hide manual ego camera frustum"
                )
                
                gui_reset_manual_ego = self.client.gui.add_button(
                    "Reset to Default Pose",
                    hint="Reset manual camera to 90° CCW rotation"
                )
                
                self.gui_manual_extrinsic_text = self.client.gui.add_markdown(
                    "**World to Manual Cam Extrinsic (4x3):**\n```\nInitializing...\n```"
                )
                
                @gui_show_manual_ego.on_update
                async def _(_) -> None:
                    if self.manual_ego_frame is not None:
                        self.manual_ego_frame.visible = gui_show_manual_ego.value
                    if self.manual_ego_frustum is not None:
                        self.manual_ego_frustum.visible = gui_show_manual_ego.value
                    if self.manual_ego_transform_handle is not None:
                        self.manual_ego_transform_handle.visible = gui_show_manual_ego.value
                
                @gui_reset_manual_ego.on_click
                def _(_) -> None:
                    self._reset_manual_ego_camera()

        await self.on_sample_update(None)

        while True:
            if (self.gui_playing is not None and self.gui_playing.value # added
                and self.gui_framerate is not None and self.gui_framerate.value > 0
            ):
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(1.0)

    async def on_sample_update(self, _):
        with self.client.atomic():
            self._rebuild_scene()
            
            # Add manual ego camera if enabled
            if self.global_context().ego_manual:
                self._add_manual_ego_camera()
                
        self._rebuild_playback_gui()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                # Use a rainbow color based on the frame index
                denom = len(self.scene_frame_handles) - 1
                rainbow_value = cm.jet(1.0 - frame_idx / denom)[:3]
                rainbow_value = tuple((int(c * 255) for c in rainbow_value))
                frame_node.frustum_handle.color = rainbow_value

    def _add_manual_ego_camera(self):
        """Add manual ego camera with transform controls."""
        logger.info("Adding manual ego camera...")
        
        default_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        default_position = np.array([0.0, 0.0, 0.0])
        default_quat = tf.SO3.from_matrix(default_rotation).wxyz
        
        # Create transform controls
        self.manual_ego_transform_handle = self.client.scene.add_transform_controls(
            "/manual_ego_camera/controls",
            scale=0.5,
            line_width=5.0,
        )
        self.manual_ego_transform_handle.wxyz = default_quat
        self.manual_ego_transform_handle.position = default_position
        
        # Create camera frustum
        fov = np.radians(60.0)
        self.manual_ego_frustum = self.client.scene.add_camera_frustum(
            "/manual_ego_camera/frustum",
            fov=fov,
            aspect=1.0,
            scale=0.2,
            color=(255, 165, 0),
            wxyz=default_quat,
            position=default_position,
        )
        
        # Create frame handle
        self.manual_ego_frame = self.client.scene.add_frame(
            "/manual_ego_camera/frame",
            wxyz=default_quat,
            position=default_position,
            axes_length=0.2,
            axes_radius=0.015,
        )
        
        # Update callback for transform controls
        @self.manual_ego_transform_handle.on_update
        def _(_) -> None:
            self._update_manual_ego_camera()
        
        # Initial update
        self._update_manual_ego_camera()
        
        logger.info("Manual ego camera added")
    
    def _reset_manual_ego_camera(self):
        """Reset manual ego camera to default pose (90° CCW around Z)."""
        if self.manual_ego_transform_handle is None:
            return
            
        # Default pose: 90° counter-clockwise rotation around Z-axis
        default_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        default_position = np.array([0.0, 0.0, 0.0])
        default_quat = tf.SO3.from_matrix(default_rotation).wxyz
        
        # Update transform control
        self.manual_ego_transform_handle.wxyz = default_quat
        self.manual_ego_transform_handle.position = default_position
        
        # Update will be triggered by on_update callback
        logger.info("Manual ego camera reset to default pose")
    
    def _update_manual_ego_camera(self):
        """Update manual ego camera frustum and extrinsic matrix display."""
        if self.manual_ego_transform_handle is None:
            return
        
        # Get current transform
        position = self.manual_ego_transform_handle.position
        wxyz = self.manual_ego_transform_handle.wxyz
        
        # Update frustum to match transform control
        if self.manual_ego_frustum is not None:
            self.manual_ego_frustum.position = position
            self.manual_ego_frustum.wxyz = wxyz
        
        # Update frame to match transform control
        if self.manual_ego_frame is not None:
            self.manual_ego_frame.position = position
            self.manual_ego_frame.wxyz = wxyz
        
        # Compute camera-to-world matrix (C2W)
        rotation_matrix = tf.SO3(wxyz).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position
        
        # Compute world-to-camera matrix (W2C, extrinsic)
        w2c = np.linalg.inv(c2w)
        w2c_4x3 = w2c[:3, :]  # Take first 3 rows (4x3 extrinsic format)
        
        # Update GUI text with formatted extrinsic matrix
        if self.gui_manual_extrinsic_text is not None:
            extrinsic_str = "**World to Manual Cam Extrinsic (4x3):**\n```\n"
            extrinsic_str += (
                f"[{w2c_4x3[0, 0]:8.4f}, {w2c_4x3[0, 1]:8.4f}, "
                f"{w2c_4x3[0, 2]:8.4f}, {w2c_4x3[0, 3]:8.4f}],\n"
            )
            extrinsic_str += (
                f"[{w2c_4x3[1, 0]:8.4f}, {w2c_4x3[1, 1]:8.4f}, "
                f"{w2c_4x3[1, 2]:8.4f}, {w2c_4x3[1, 3]:8.4f}],\n"
            )
            extrinsic_str += (
                f"[{w2c_4x3[2, 0]:8.4f}, {w2c_4x3[2, 1]:8.4f}, "
                f"{w2c_4x3[2, 2]:8.4f}, {w2c_4x3[2, 3]:8.4f}]\n"
            )
            extrinsic_str += "```"
            self.gui_manual_extrinsic_text.content = extrinsic_str


    def _rebuild_scene(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        spatial_subsample: int = self.gui_s_sub.value
        temporal_subsample: int = self.gui_t_sub.value
        
        # GlobalContext에서 설정값 가져오기
        use_mean_bg = self.global_context().use_mean_bg

        rays: np.ndarray | None = None
        first_frame_y: np.ndarray | None = None

        self.client.scene.reset()
        
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []
        
        # Mean background depth 계산 (use_mean_bg가 true이고 아직 계산되지 않은 경우)
        if use_mean_bg and (self.global_context().mean_depth is None or self.global_context().mean_rgb is None):
            logger.info("Computing mean background depth for the first time...")
            mean_depth, mean_rgb = compute_mean_background_depth(
                current_artifact, spatial_subsample
            )
            # GlobalContext에 저장
            self.global_context().mean_depth = mean_depth
            self.global_context().mean_rgb = mean_rgb

        def none_it(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None

        # 마스크 데이터 로더 추가
        def none_it_mask(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None

        for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth), (_, instance_mask)) in enumerate(
            zip(
                read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy(),
                read_rgb_artifacts(current_artifact.rgb_path),
                *read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1:3],
                none_it(read_depth_artifacts(current_artifact.depth_path)),
                none_it_mask(read_instance_artifacts(current_artifact.mask_path)),
            )
        ):
            if frame_idx % temporal_subsample != 0:
                continue

            # Current frame size
            frame_height, frame_width = rgb.shape[:2]

            # Use ViPE intrinsics
            pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())

            # RGB 처리 - use_mean_bg에 따라 다른 RGB 사용
            if not (use_mean_bg and self.global_context().mean_rgb is not None):
                # 일반적인 프레임별 RGB 사용
                sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
                sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

            if first_frame_y is None:
                first_frame_y = c2w[:3, 1]
                self.client.scene.set_up_direction(-first_frame_y)

            if rays is None:
                camera_model = camera_type.build_camera_model(intr)
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                if camera_type == CameraType.PANORAMA:
                    disp_v = disp_v / (frame_height - 1)
                    disp_u = disp_u / (frame_width - 1)
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                if camera_type != CameraType.PANORAMA:
                    rays /= rays[..., 2:3]

            # Point cloud 처리 - use_mean_bg에 따라 다른 방식 사용
            dynamic_pcd = None
            dynamic_colors = None
            
            if use_mean_bg and self.global_context().mean_depth is not None:
                # Mean background depth 사용
                logger.info(f"Frame {frame_idx}: Using pre-computed mean background depth with nanmean")
                
                # GPU tensor를 numpy로 변환
                mean_depth_np = self.global_context().mean_depth.cpu().numpy()
                mean_rgb_np = self.global_context().mean_rgb.cpu().numpy()
                
                # Background point cloud 계산 (mean depth 사용)
                pcd = rays * mean_depth_np[..., None]
                
                # Mean depth에서 유효한 픽셀 마스크 (0이 아닌 값들)
                depth_mask = mean_depth_np > 0
                
                # RGB 색상도 mean RGB 사용 (0-255 범위로 변환)
                sampled_rgb = (mean_rgb_np * 255).astype(np.uint8)
                
                # Dynamic objects 추가 (현재 프레임의 실제 depth 사용)
                show_dynamic = True  # 기본값
                if hasattr(self, 'gui_show_dynamic_objects') and self.gui_show_dynamic_objects is not None:
                    show_dynamic = self.gui_show_dynamic_objects.value
                
                if depth is not None and instance_mask is not None and show_dynamic:
                    dynamic_pcd, dynamic_colors = build_dynamic_points_for_frame(
                        current_artifact, frame_idx, c2w, rgb, depth, instance_mask, 
                        camera_type, intr, spatial_subsample
                    )
                
                logger.info(f"Frame {frame_idx}: Using mean background with {np.sum(depth_mask)} background points + {len(dynamic_pcd) if dynamic_pcd is not None else 0} dynamic points")
                
            elif depth is not None:
                # 원래 코드 (프레임별 depth 사용)
                pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]                
                depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
                
                # 인스턴스 마스크를 이용한 동적 객체 필터링 (GUI 설정에 따라)
                if instance_mask is not None and self.gui_filter_dynamic_objects.value:
                    # 인스턴스 마스크를 numpy로 변환
                    instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
                    # 배경(0)만 유지하고 동적 객체(1=person, 2=car 등) 필터링
                    static_mask = (instance_mask_np == 0)
                    # 공간 서브샘플링 적용
                    static_mask_sub = static_mask[::spatial_subsample, ::spatial_subsample]
                    # 깊이 마스크와 정적 마스크 결합
                    depth_mask = depth_mask & static_mask_sub
                    logger.info(f"Frame {frame_idx}: Applied instance mask filtering (kept {np.sum(depth_mask)} / {depth_mask.size} points)")
            else:
                pcd, depth_mask = None, None

            frame_node = self._make_frame_nodes(
                frame_idx,
                c2w,
                sampled_rgb,
                fov,
                pcd,
                depth_mask,
                dynamic_pcd,
                dynamic_colors,
            )
            self.scene_frame_handles.append(frame_node)

    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd: np.ndarray | None,
        pcd_mask: np.ndarray | None = None,
        dynamic_pcd: np.ndarray | None = None,
        dynamic_colors: np.ndarray | None = None,
    ) -> SceneFrameHandle:
        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=0.05,
            axes_radius=0.005,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )
        frame_height, frame_width = rgb.shape[:2]

        frame_thumbnail = Image.fromarray(rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=frame_width / frame_height,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        # Background point cloud 처리
        if pcd is not None:
            pcd = pcd.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            if pcd_mask is not None:
                pcd_mask = pcd_mask.reshape(-1)
                pcd = pcd[pcd_mask]
                rgb = rgb[pcd_mask]
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud_bg",
                points=pcd,
                colors=rgb,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        else:
            pcd_handle = None

        # Dynamic point cloud 처리
        dynamic_pcd_handle = None
        if dynamic_pcd is not None and len(dynamic_pcd) > 0:
            # Dynamic objects를 카메라 좌표계에서 월드 좌표계로 변환
            dynamic_world = (c2w[:3, :3] @ dynamic_pcd.T + c2w[:3, 3:4]).T
            
            dynamic_pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud_dynamic",
                points=dynamic_world,
                colors=dynamic_colors,
                point_size=self.gui_point_size.value * 1.2,  # Dynamic objects를 조금 더 크게
                point_shape="rounded",
            )
            logger.info(f"Frame {frame_idx}: Added {len(dynamic_world)} dynamic points to scene")

        return SceneFrameHandle(
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
            dynamic_pcd_handle=dynamic_pcd_handle,
        )

    def _incr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _rebuild_playback_gui(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        self.gui_name.value = current_artifact.artifact_name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=len(self.scene_frame_handles) - 1, step=1, initial_value=0
            )
            gui_frame_control = self.client.gui.add_button_group("Control", options=["Prev", "Next"])
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=15)
            #### added ####
            self.gui_playing = self.client.gui.add_checkbox("Playing", initial_value=False)
            self.gui_show_all_frames = self.client.gui.add_checkbox("Show all frames", initial_value=False)

            @self.gui_show_all_frames.on_update
            async def _(_):
                # 토글 켜면 모든 프레임 보이기, 끄면 타임라인 위치만 보이기
                show_all = self.gui_show_all_frames.value
                with self.client.atomic():
                    if show_all:
                        for h in self.scene_frame_handles:
                            h.visible = True
                        # Show all ego frustums
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            for ego_frustum in self.ego_frustum_handles:
                                ego_frustum.visible = True
                    else:
                        for i, h in enumerate(self.scene_frame_handles):
                            h.visible = (i == self.gui_timestep.value)
                        # Show only current frame's ego frustum
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            for i, ego_frustum in enumerate(self.ego_frustum_handles):
                                ego_frustum.visible = (i == self.gui_timestep.value)
            ###############

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "Prev":
                    self._decr_timestep()
                else:
                    self._incr_timestep()

            self.current_displayed_timestep = self.gui_timestep.value

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_timestep = self.gui_timestep.value
                prev_timestep = self.current_displayed_timestep
                with self.client.atomic():
                    if not (self.gui_show_all_frames and self.gui_show_all_frames.value): # added
                        self.scene_frame_handles[current_timestep].visible = True
                        self.scene_frame_handles[prev_timestep].visible = False
                        
                        # Update ego camera frustum visibility
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            # Hide all ego frustums first
                            for ego_frustum in self.ego_frustum_handles:
                                ego_frustum.visible = False
                            # Show only current frame's ego frustum
                            if current_timestep < len(self.ego_frustum_handles):
                                self.ego_frustum_handles[current_timestep].visible = True
                self.current_displayed_timestep = current_timestep

    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def get_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def run_viser(base_path: Path, port: int = 20540, use_mean_bg: bool = False, ego_manual: bool = False):
    # Get list of artifacts.
    logger.info(f"Loading artifacts from {base_path}")
    artifacts: list[ArtifactPath] = list(ArtifactPath.glob_artifacts(base_path, use_video=True))
    if len(artifacts) == 0:
        logger.error("No artifacts found. Exiting.")
        return

    global _global_context
    _global_context = GlobalContext(
        artifacts=sorted(artifacts, key=lambda x: x.artifact_name),
        use_mean_bg=use_mean_bg,
        ego_manual=ego_manual
    )

    # 새 코드: 모든 인터페이스에서 수신 대기
    server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
    # 원래 코드 (주석 처리)
    # server = viser.ViserServer(host=get_host_ip(), port=port, verbose=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        # wait synchronously in this function for task to be finished.
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Visualizer")
    parser.add_argument("base_path", type=Path, help="Base path for the visualizer")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=20540,
        help="Port number for the viser server.",
    )
    parser.add_argument(
        "--use_mean_bg", 
        action="store_true", 
        help="Use nanmean background instead of standard background"
    )
    parser.add_argument(
        "--ego_manual",
        action="store_true",
        help="Enable manual ego camera control with transform handles"
    )
    args = parser.parse_args()

    run_viser(args.base_path, args.port, args.use_mean_bg, args.ego_manual)


if __name__ == "__main__":
    main()
