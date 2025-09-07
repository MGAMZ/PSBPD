import math, random, pdb

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_hist
from skimage.transform import warp
from skimage.transform import PiecewiseAffineTransform

from mmcv.transforms.base import BaseTransform
import cv2


def rectangular_to_polar(x, y, center_x, center_y):
    """
    Rectangular coordinates start from 0
    Standard rectangular coordinate system input: x, y
    Rectangular coordinates of the pole: center_x, center_y
    
    radius: polar radius
    angle: polar angle in radians
    """
    # Use numpy to compute radius
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Use scipy to compute angle
    angle = np.arctan2(y - center_y, x - center_x)
    
    return radius, angle


def polar_to_rectangular(radius, angle, center_x, center_y):
    """
    Rectangular coordinates start from 0
    radius: polar radius
    angle: polar angle in radians
    center_x, center_y: rectangular coordinates of the pole

    x, y: rectangular coordinates
    """
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    
    return x, y


class ScanTableRemover(BaseTransform):
    def __init__(self, pad_val=-1024, TableIntrinsicCircleRadius=600, MaskOffset=0, pixel_array_shape=(512,512)):
        if len(pixel_array_shape) == 1:
            self.pixel_array_shape = pixel_array_shape
        elif len(pixel_array_shape) == 2:
            assert pixel_array_shape[0] == pixel_array_shape[1], "pixel_array_shape should be a square"
            self.pixel_array_shape = pixel_array_shape[0]
        else:
            raise ValueError("Only 2D array is supported")
        self.pad_val = pad_val
        self.TableIntrinsicCircleRadius = TableIntrinsicCircleRadius
        self.MaskOffset = MaskOffset
        self.PixelSpacing_offset = 1

    def process(self, pixel_array:np.ndarray, table_height:float, 
                recon_center_cord:tuple[float,float], 
                pixel_spacing:float|np.ndarray):
        if len(pixel_spacing) == 2:	# type:ignore
            if pixel_spacing[0] == pixel_spacing[1]:	# type:ignore
                pixel_spacing = pixel_spacing[0]	# type:ignore
            else:
                raise ValueError("If 2D array, only square shape is supported")
        pixel_spacing = pixel_spacing * self.PixelSpacing_offset

        # Calculate distance of various elements in the vertical center of the slice
        Position_MaskCenter = table_height + self.MaskOffset - self.TableIntrinsicCircleRadius
        Distance_MaskCenter_ReconCenter = recon_center_cord[0] - Position_MaskCenter

        # Physical unit (mm) -> Image unit (pixel)
        PixelCord_MaskCenter = (self.pixel_array_shape//2 - Distance_MaskCenter_ReconCenter/pixel_spacing,
                                self.pixel_array_shape//2 - recon_center_cord[1]           /pixel_spacing)
        PixelDistance_MaskRadius = self.TableIntrinsicCircleRadius / pixel_spacing

        # print(f"Pixel Mask Param: center {PixelCord_MaskCenter}, radius {PixelDistance_MaskRadius}")
        # print(f"Pixel Dist Param: MaskCenter_ReconCenter {Distance_MaskCenter_ReconCenter} ReconCenter_ScanBed {Distance_ReconCenter_ScanBed} MaskCenter_ScanBed {Distance_MaskCenter_ScanBed}")
        # Execute mask
        for x in range(self.pixel_array_shape):
            for y in range(self.pixel_array_shape):
                if (x-PixelCord_MaskCenter[0])**2+(y-PixelCord_MaskCenter[1])**2 > PixelDistance_MaskRadius**2:
                    pixel_array[x][y] = self.pad_val
        
        return pixel_array


    def transform(self, results: dict) -> dict:
        # Some cases without complete dcm sequences cannot perform bed removal 
        # because their dependent metadata does not exist.
        if results['dcm_meta']:
            results['img'] = self.process(
                pixel_array=results['img'], 
                table_height=results['dcm_meta']['00181130']['Value'][0], 
                recon_center_cord=results['dcm_meta']['00431031']['Value'], 
                pixel_spacing=results['dcm_meta']['00280030']['Value']
            )
        return results


class Distortion(BaseTransform):
    def __init__(self,
                 global_rotate:float,
                 amplitude:float,
                 frequency:float,
                 grid_dense:int,
                 in_array_shape:tuple,
                 refresh_interval:int|None=None,
                 pad_val:int=-1024,
                 seg_pad_val:int=0,
                 use_cv2:bool=True,
                 const:bool=False,
                 ) -> None:
        self.global_rotate = global_rotate  # Random global rotation
        self.amplitude = amplitude  # Amplitude
        self.frequency = frequency  # Frequency
        self.img_shape = in_array_shape  # Size of input matrix
        self.grid_dense = grid_dense  # Grid density
        self.refresh_interval = refresh_interval  # Mapping matrix refresh interval
        self.refresh_counter = 0
        self.const = const # Control whether to fix AF parameters
        self.use_cv2 = use_cv2
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        super().__init__()

    # Generate mapping matrix for piecewise affine transformation
    # Sometimes to fix AF parameters, const is used to disable random parameters
    def refresh_affine_map(self, const:bool):
        src_cols = np.linspace(0, self.img_shape[0], self.grid_dense)
        src_rows = np.linspace(0, self.img_shape[1], self.grid_dense)
        # Build grid, src_rows, src_cols have shape (grid_dense, grid_dense)
        # src_rows are row coordinates of grid points in source image
        # src_cols are column coordinates of grid points in source image
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        # src contains coordinates of all grid points in source image, shape (grid_dense, grid_dense, 2)
        src = np.stack([src_cols, src_rows], axis=2)
        dst = np.zeros_like(src)

        amplitude = (1 if const else random.random()*2-1) * self.amplitude
        frequency = (1 if const else random.random()*2-1) * self.frequency
        global_rotate = (1 if const else random.random()*2-1) * self.global_rotate
        
        for x in range(self.grid_dense):
            for y in range(self.grid_dense):
                # Convert index to standard coordinates
                y = self.grid_dense - y - 1
                # Convert rectangular to polar coordinates
                radius, angle = rectangular_to_polar(x, y, self.grid_dense//2, self.grid_dense//2)
                # Convert mapping grid coordinate system to source coordinate system
                radius *= self.img_shape[0] / self.grid_dense
                # Radial distortion
                angle += math.pi/8 * amplitude * np.sin(radius/self.img_shape[0] * frequency * 2 * math.pi)
                # Global rotation
                angle += global_rotate * math.pi/2
                # Convert polar to rectangular coordinates
                src_x, src_y = polar_to_rectangular(radius, angle, self.img_shape[0]/2, self.img_shape[1]/2)
                # Convert standard coordinates to index
                src_y = self.img_shape[0] - src_y - 1
                y = self.grid_dense - y - 1
                # Store in dst
                dst[x, y, :] = (src_x, src_y)

        def calc_cv2_map(img_shape, tform):
            src = np.mgrid[0:img_shape[0], 0:img_shape[1]].astype(np.float32)
            src = src.transpose(1,2,0)  # (H,W,2)
            src = src.reshape(img_shape[0]*img_shape[1], 2) # (H*W,2)
            dst = tform(src)
            dst = dst.reshape(img_shape[0], img_shape[1], 2) # (H,W,2)
            cv2_mapY = dst[..., 0].astype(np.float32)
            cv2_mapX = dst[..., 1].astype(np.float32)
            return cv2_mapY, cv2_mapX
        
        tform = PiecewiseAffineTransform()
        tform.estimate(src.reshape(-1,2), dst.reshape(-1,2))
        cv2_mapY, cv2_mapX = calc_cv2_map(self.img_shape, tform)
        
        return tform, (cv2_mapX, cv2_mapY)

    @staticmethod
    def distort(tform, Imgarray:np.ndarray, order, pad_val):
        return warp(image=Imgarray,
                    inverse_map=tform,
                    order=order,
                    preserve_range=True,
                    cval=pad_val)

    @staticmethod
    def distort_cv2(Imgarray:np.ndarray,
                    map1:np.ndarray,
                    map2:np.ndarray,
                    pad_val:int,
                    interpolation):
        return cv2.remap(src=Imgarray,
                         map1=map1.astype(np.float32),
                         map2=map2.astype(np.float32),
                         interpolation=interpolation,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=pad_val)

    def transform(self, results: dict) -> dict:
        # 在开始时或每隔一段时间，刷新映射矩阵
        if (not self.const
            and self.refresh_interval is not None
            and self.refresh_counter % self.refresh_interval == 0
        ):
            self.tform, (self.cv2_map1, self.cv2_map2) = self.refresh_affine_map(
                self.img_shape,
                self.grid_dense,
                self.amplitude,
                self.frequency,
                self.global_rotate,
                self.const)
        
        if self.use_cv2:
            results['img'] = self.distort_cv2(
                results['img'], 
                self.cv2_map1, 
                self.cv2_map2, 
                self.pad_val,
                interpolation=cv2.INTER_CUBIC)
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = self.distort_cv2(
                    results['gt_seg_map'], 
                    self.cv2_map1, 
                    self.cv2_map2, 
                    self.seg_pad_val,
                    interpolation=cv2.INTER_NEAREST)
        else:
            results['img'] = self.distort(
                tform=self.tform,
                Imgarray=results['img'],
                order=1,
                pad_val=self.pad_val)
            if 'gt_seg_map' in results:
                results['gt_seg_map'] = self.distort(
                    tform=self.tform,
                    Imgarray=results['gt_seg_map'],
                    order=0,
                    pad_val=self.seg_pad_val)
        
        self.refresh_counter += 1
        return results

# Original Images min: -2,000, max: 4,095, mean: 10.87, std: 1,138.37
# after clip: min: 0, max: 4,095, mean: 561.54, std: 486.59
class RangeClipNorm(BaseTransform):
    def __init__(self, input_shape, norm_method:str, mean=None, std=None) -> None:
        assert norm_method in ['const', 'inst', 'hist'], "[Dataset] CTImageEnhance Augmentation: norm_method must be 'const' or 'inst'"
        if mean is None or std is None: 
            assert norm_method!='const', "[Dataset] CTImageEnhance Augmentation: mean and std must be provided when norm_method is 'const'"
            self.mean = self.std = None
        self.norm_method = norm_method

        if norm_method=='hist':
            self.nbins = 256
            self.mask = self.create_circle_in_square(input_shape[0], input_shape[0]//3)

        self.input_shape = input_shape
        super().__init__()

    @staticmethod
    def create_circle_in_square(size, radius):
        # Create a square ndarray filled with zeros
        square = np.zeros((size, size))
        # Calculate center point coordinates
        center = size // 2
        # Calculate distance from each element to center
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        # If distance is less than or equal to radius, set element to 1
        square[mask] = 1
        return square


    def hist_equal(self, Imgarray:np.ndarray):
        assert Imgarray.shape == self.input_shape, f"HistogramEqualization Augmentation: input_shape expected{self.input_shape}, got {Imgarray.shape}"
        return equalize_hist(Imgarray, nbins=self.nbins, mask=self.mask)

    # Custom algorithm
    def _exec(self, Imgarray:np.ndarray):
        Imgarray = np.clip(Imgarray, -1024, 4096).astype(np.float32)
        # Normalize
        if self.norm_method == 'const':
            Imgarray = (Imgarray - self.mean) / self.std
        elif self.norm_method == 'inst':
            Imgarray = (Imgarray - Imgarray.mean()) / Imgarray.std()
        elif self.norm_method == 'hist':
            Imgarray = equalize_hist(Imgarray, nbins=self.nbins, mask=self.mask)

        return Imgarray
    
    # MMSegmentation interface
    def transform(self, results: dict) -> dict:
        ImgNdarray = results['img']
        assert isinstance(ImgNdarray, np.ndarray), "CTImageEnhance Augmentation: input img must be a numpy ndarray"
        ImgNdarray = self._exec(ImgNdarray)
        # print('enhenced:', ImgNdarray.shape, ImgNdarray.min(), ImgNdarray.max(), ImgNdarray.mean(), ImgNdarray.std(), ImgNdarray.dtype)
        results['img'] = ImgNdarray
        results['img_shape'] = ImgNdarray.shape[:2]
        return results


class GaussianBlur(BaseTransform):
    def __init__(self, sigma, radius) -> None:
        self.sigma = sigma
        self.radius = radius
        super().__init__()
    
    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        results['img'] = gaussian_filter(results['img'], 
                                         sigma=self.sigma, 
                                         order=3,
                                         radius=self.radius,
                                         )
        return results


class OriShapeOverride(BaseTransform):
    def __init__(self, ori_shape:tuple) -> None:
        super().__init__()
        self.ori_shape = ori_shape

    def transform(self, results: dict) -> dict:
        results['img_shape'] = self.ori_shape
        results['ori_shape'] = self.ori_shape
        results['scale_factor'] = (1,1)
        return results


class ConfirmShape_HWC(BaseTransform):
    def __init__(self, check_img:bool=True, check_anno:bool=True):
        self.check_img = check_img
        self.check_anno = check_anno
    
    def transform(self, results: dict):
        if results['img'].ndim == 2 and self.check_img:
            results['img'] = results['img'][..., np.newaxis]
        if results['gt_seg_map'].ndim == 2 and self.check_anno:
            results['gt_seg_map'] = results['gt_seg_map'][..., np.newaxis]
        
        return results


class LabelResize(BaseTransform):
    def __init__(self, size):
        self.size = size
    def transform(self, results):
        results['gt_seg_map'] = cv2.resize(
            results['gt_seg_map'], 
            self.size, 
            interpolation=cv2.INTER_NEAREST)
        return results
