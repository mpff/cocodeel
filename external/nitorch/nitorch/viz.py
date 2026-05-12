# @Author: JiHoon Kim, jihoon.kim@fu-berlin.de
# Created: 15.05.2022
# Last updated: 15.11.2022
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.image import resample_img
from math import gcd
import ipywidgets as widgets
from ipywidgets import fixed, Button, interact
from IPython.display import display


def plot_mri(
    image,
    zoom:float=1,
    interpolation:str='nearest',
    figsize:tuple=(20,5),
    title:str=" ",
    cmap="gray",
    axis:bool =False,
    header:bool=False,
    grid:bool=False
    ):
    """ Interactive MRI visualizer, RAS+ coordinate systems (optional: support reshape)
    Parameters
    ----------
    image: nib.Nifti1Image | str | np.ndarray
        mri data
    zoom: float, optional
        zoom level, zoom <1 upsampled; zoom >1 downsampled image. Default: 1
        https://nilearn.github.io/dev/modules/generated/nilearn.image.resample_img.html
    interpolation: str, optional
        Can be ‘continuous’, ‘linear’, or ‘nearest’. Indicates the resample method. Default: 'nearest'
        https://nilearn.github.io/dev/modules/generated/nilearn.image.resample_img.html
    figsize: tuple, optional
        figure size (recommended ratio = 3:1). Default: (15,5)
    title: str, optional
        suptitle heading. Default: " "
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html
    cmap: str or Colormap, optional
        The Colormap instance or registered colormap name used to map scalar data to colors. Default: "gray"
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    axis: boolean, optional
        If True, Display the axis of the i, j, k. Default: False
    header: boolean, optional
        If True, Display the image hedaer info bar. Default: False
        https://nipy.org/nibabel/nifti_images.html#the-nifti-header
    grid: boolean, optional
        If True, Display the voxel grid using gcd. Default: True
        https://docs.python.org/3/library/math.html#math.gcd
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
        
    Returns
    -------
    data: nib.Nifti1Image
        preprocessed image on RAS+ coordinate system with reshape with zoom

    Notes
    -----
    RAS+
        https://nipy.org/nibabel/coordinate_systems.html
    Ipywidgets
        https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html#More-control-over-the-user-interface:-interactive_output
    
    If you have any problems, please contact Jihoon, jihoon.kim@fu-berlin.de
    
    (TODO)
    ------
    - Set the voxel values distribution (showing only non-zeros) parameter
    - Update the xset yset size check
    - Update the zoom interpolation func using masked MNI
    - Segmentation viz supported, based on mask gray, white, csf matter.
    """
    # 1. Check the data type
    if isinstance(image, str) and os.path.isfile(image):
        data = nib.load(image)
    elif isinstance(image, nib.Nifti1Image):
        data = image
    elif isinstance(image, np.ndarray) and len(image.shape) == 3:
        data = nib.Nifti1Image(image, affine=np.eye(4))
        print("CAUTION: Only for visualization. Please check the affine and header.")
    else:
        raise TypeError(f"Invalid type: {type(image)}, IF AND ONLY IF either valid file path or Nifti1Image")
    # 2. Reorient the RAS+ coordinate system
    data = nib.as_closest_canonical(data)
    
    # 3. Reshape the data
    if zoom:
        data = resample_img(data, target_affine=data.affine*zoom,
                            interpolation=interpolation)
    
    # 4. Load data and header
    vol = data.get_fdata()
    hdr = data.header
    RAS = nib.aff2axcodes(data.affine)
    sform = ''.join(RAS)+"+"
    vals = vol.ravel()
    vals = vals[vals != 0]
    
    # 5. Set the volume i, j, k range
    iv, jv, kv = vol.shape
    # i,j,k = vol.shape
    
    # 6. Compute gcd
    N = gcd(iv,jv,kv)
    divisor = []
    for n in range(2, int(N ** 0.5) + 1):
    	if N % n == 0:
    		divisor.append(n)
    		if N // n != n:
    			divisor.append(N // n)
    divisor.append(N)
    divisor.sort()       
    
    # 7. Set the widgets
    wid_i = widgets.IntSlider(min=0, max=iv-1)
    wid_j = widgets.IntSlider(min=0, max=jv-1)
    wid_k = widgets.IntSlider(min=0, max=kv-1)
    wid_g = widgets.Dropdown(options=divisor, value=divisor[0], description='Grid size:')
    ui = widgets.HBox([wid_i, wid_j, wid_k, wid_g])
    
    # 8. Set the default values
    wid_i.default_value = int(iv * 0.38)
    wid_j.default_value = int(jv * 0.38)
    wid_k.default_value = int(kv * 0.45)
    wid_g.default_value = int(divisor[len(divisor)//2])

    defaulting_widgets = [wid_i, wid_j, wid_k, wid_g]
    default_value_button = Button(description='Click to set default')

    def set_default(button):
        for widget in defaulting_widgets:
            widget.value = widget.default_value

    default_value_button.on_click(set_default)
    
    # 9. Plot the Brain MRI
    def f(i,j,k,g):
        # line and width size
        (f, (ax_yz, ax_xz, ax_xy, ax_di)) = plt.subplots(1,4, figsize=figsize)
        hd_size = figsize[0]
        subhd_size = figsize[1]*2.5
        annot_size = figsize[1]*2
        ref_size = figsize[1]*1.5
        line_width = figsize[1]*0.3
        grid_width = figsize[1]*0.2

        # (1,1) R+
        ax_yz.set_title(f"Sagittal cross-section at $i$={i}", fontsize=subhd_size)
        ax_yz.imshow(vol[i,:,:].T, cmap=cmap, origin='lower')
        ax_yz.set_xlabel("Index $j$ (P $\mapsto$ A)", fontsize=annot_size)
        ax_yz.set_ylabel("Index $k$ (I $\mapsto$ S)", fontsize=annot_size)
        # (1,2) A+ 
        ax_xz.set_title(f"Coronal cross-section at $j$={j}", fontsize=subhd_size)
        ax_xz.imshow(vol[:,j,:].T,   cmap=cmap, origin='lower')
        ax_xz.set_xlabel('Index $i$ (L $\mapsto$ R)', fontsize=annot_size)
        ax_xz.set_ylabel('Index $k$ (I $\mapsto$ S)', fontsize=annot_size)
        # (1,3) S+
        ax_xy.set_title(f"Axial cross-section at $k$={k}", fontsize=subhd_size)
        ax_xy.imshow(vol[:,:,k].T,  cmap=cmap, origin='lower')
        ax_xy.set_xlabel('Index $i$ (L $\mapsto$ R)', fontsize=annot_size)
        ax_xy.set_ylabel('Index $j$ (P $\mapsto$ A)', fontsize=annot_size)
        # (1,4) Distribution
        ax_di.set_title("voxel values distribution (showing only non-zeros)")
        ax_di.hist(vals, bins=20)
        
        # axis
        if axis:
            ax_yz.axvline(x=j,color='red', linewidth=line_width)
            ax_yz.axhline(y=k,color='red', linewidth=line_width)
            ax_xz.axvline(x=i,color='red', linewidth=line_width)
            ax_xz.axhline(y=k,color='red', linewidth=line_width)
            ax_xy.axvline(x=i,color='red', linewidth=line_width)
            ax_xy.axhline(y=j,color='red', linewidth=line_width)
        
        # grid
        if (grid and g!=0):
            # R+
            ax_yz.set_xticks(np.arange(0, jv, jv/g))
            ax_yz.set_yticks(np.arange(0, kv, kv/g))
            ax_yz.grid(True, axis='x', color='red', linewidth=grid_width, linestyle='--')
            ax_yz.grid(True, axis='y', color='red', linewidth=grid_width, linestyle='--')
            # A+
            ax_xz.set_xticks(np.arange(0, iv, iv/g))
            ax_xz.set_yticks(np.arange(0, kv, kv/g))
            ax_xz.grid(True, axis='x', color='red', linewidth=grid_width, linestyle='--')
            ax_xz.grid(True, axis='y', color='red', linewidth=grid_width, linestyle='--')
            # S+
            ax_xy.set_xticks(np.arange(0, iv, iv/g))
            ax_xy.set_yticks(np.arange(0, jv, jv/g))
            ax_xy.grid(True, axis='x', color='red', linewidth=grid_width, linestyle='--')
            ax_xy.grid(True, axis='y', color='red', linewidth=grid_width, linestyle='--')    

        f.suptitle(f"{title}",fontsize = figsize[0])# 'xx-large'
        plt.figtext(0.08, 0.02, f"Data orientation {sform}\
        \nData shape {(iv, jv, kv)}", fontsize = ref_size)
    out = widgets.interactive_output(f, {'i':wid_i, 'j':wid_j, 'k':wid_k, 'g':wid_g})
    display(ui, out, default_value_button)
    
    def view_header(header):
        return print(hdr[header])
    if header:
        interact(view_header, header=(hdr))
    return data



# def plot_dataset(plot_X, plot_y, idx, **args):
#     """ Interactive dataset viewer
#     Parameters
#     ----------
#     X: np.ndarray
#         set of mri
#     y: np.ndarray
#         set of labels
#     idx: int
#         index number of data sampled
#     **args: dict, optional
#         parameters of the plot_mri, except title
    
#     Returns
#     -------
#     None
    
#     (TODO) 
#     ------
#     Support the meta data info
#     Flexible title support with y
#     """
    
#     # plot_mri of the random N data
#     def f(N):
#         X_i, y_i = plot_X[N], plot_y[N]
#         title = f"Data X sampled at N={N} (y={y_i})"
#         return plot_mri(X_i, title=title, **args)
#     # run the interactive widget
#     interact(f, N = widgets.IntSlider(min=0, max=len(plot_y)-1, step=1, value=idx))
    

# (TODO): pairwise dataset widget
    
# (TODO): batch size viz

# (TODO): 3D plot viz
