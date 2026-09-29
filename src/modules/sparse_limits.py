"""Allocation-free checks for spconv kernels using int32 byte offsets."""


def check_spconv_feature_size(
    voxel_count: int,
    channels: int,
    element_size: int,
    *,
    stage: str,
) -> None:
    """Reject feature matrices which violate the kernel's strict byte limit.

    This checks one feature matrix, not total GPU memory or convolution
    workspaces. A smaller matrix is not a guarantee that the full model fits.
    """
    byte_count = int(voxel_count) * int(channels) * int(element_size)
    int32_max = (1 << 31) - 1
    if byte_count >= int32_max:
        raise RuntimeError(
            "spconv data exceed int32 range before allocation: "
            f"stage={stage}, voxels={voxel_count}, channels={channels}, "
            f"element_bytes={element_size}, feature_bytes={byte_count}, "
            f"required_feature_bytes<{int32_max}. "
            "Use a coarser projection grid to reduce the retained voxel count."
        )
