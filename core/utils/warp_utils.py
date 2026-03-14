import torch
import torch.nn as nn
import inspect


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid, full_padded_img_h=None, full_padded_img_w=None):
    if full_padded_img_h is None or full_padded_img_w is None:
        _, _, H, W = v_grid.size()
    else:
        H, W = full_padded_img_h, full_padded_img_w

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)

def position_plus_flow(flow): # get corresponding position based on a flow field. flow: B 2 h w
    """Compute the warp from the flow field.

    Args:
        flow: tf.tensor representing optical flow.

    Returns:
        The warp, i.e. the endpoints of the estimated flow.
    """
    batch, channels, ht, wd = flow.shape
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(batch, 1, 1, 1)

    position_plus_flow = coords.to(flow.device) + flow # test of this makes sense!!
    return position_plus_flow


def mask_out_of_image(coords, pad_h=0, pad_w=0, max_height=None , max_width=None): # coords B, 2, h, w
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.
    pad_h: int, the amount of padding applied to the top of the image
    pad_w: int, the amount of padding applied to the left of the image

  Returns:
    The mask showing which coordinates are valid.
  """
  pad_h = float(pad_h)
  pad_w = float(pad_w)
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  if max_height is None:
    max_height = float(coords.shape[2] - 1)
  if max_width is None:
    max_width = float(coords.shape[3] - 1)

  mask = torch.logical_and(
      torch.logical_and(coords[:, 0, :, :] >= pad_w,
                     coords[:, 0, :, :] <= max_width),
      torch.logical_and(coords[:, 1, :, :] >= pad_h,
                     coords[:, 1, :, :] <= max_height))
  
  mask = mask.type(torch.float).unsqueeze(dim=1)
  return mask


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = flow12.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12, full_padded_img_h=x.shape[2], full_padded_img_w=x.shape[3])  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons

def get_guassian_consistency_mask(flow, flow_bw, sigma=0.03):# flow: list of batch. If teacher, then just a batch
    is_list = False
    flow_may_be_list = flow
    if isinstance(flow, list):
        is_list = True
        flow=torch.cat(flow, dim=0)
        flow_bw=torch.cat(flow_bw, dim=0)
    h, w = flow.shape[2], flow.shape[3]
    #compute out of image flows:
    corresponding_pixels = position_plus_flow(flow)
    out_of_img_mask = mask_out_of_image(corresponding_pixels)
    #compute fw bw consistency
    flow_bw_warped = flow_warp(flow_bw, flow, pad='zeros')
    flow_fw_bw_sq_diff = torch.unsqueeze(torch.sum((flow + flow_bw_warped)**2, dim=1), dim=1)
    fw_bw_consistency = torch.exp(-flow_fw_bw_sq_diff / (((sigma)**2)*(h**2 + w**2)))
    flow_validity_mask = fw_bw_consistency * out_of_img_mask
    if is_list:
        flow_validity_mask = torch.chunk(flow_validity_mask, len(flow_may_be_list))

    return flow_validity_mask


def get_occu_mask_bidirection(flow12_list, flow21_list, scale=0.01, bias=0.5):
    flows12=torch.cat(flow12_list, dim=0).detach()
    flows21=torch.cat(flow21_list, dim=0).detach()
    
    flow21_warped = flow_warp(flows21, flows12, pad='zeros')
    flow12_diff = flows12 + flow21_warped
    mag = (flows12 * flows12).sum(1, keepdim=True) + \
        (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occs = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    occs = 1 - occs.float()
    corresponding_pixels = position_plus_flow(flows12)
    out_of_img_mask = mask_out_of_image(corresponding_pixels)
    occs = occs * out_of_img_mask
    occs = torch.chunk(occs, len(flow12_list)) # occ:0, noc:1

    return occs


def get_occu_mask_backward(flow12_list, flow21_list, th=0.2):
    flows12=torch.cat(flow12_list, dim=0)
    flows21=torch.cat(flow21_list, dim=0)
    B, _, H, W = flows21.size()
    base_grid = mesh_grid(B, H, W).type_as(flows21)  # B2HW
    corr_map = get_corresponding_map(base_grid + flows21)  # BHW
    occs = corr_map.clamp(min=0., max=1.)
    corresponding_pixels = position_plus_flow(flows12)
    out_of_img_mask = mask_out_of_image(corresponding_pixels)
    occs = occs * out_of_img_mask
    occs = occs.detach()
    occs = torch.chunk(occs, len(flow12_list)) # occ:0, noc:1

    return occs

