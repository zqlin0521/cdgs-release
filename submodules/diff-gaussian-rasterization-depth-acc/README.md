# Differential Gaussian Rasterization with Depth and Accuracy Map

[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
[Rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d)

## Install
```
pip install <path of this repo> -e
```

At `gaussian_renderer/__init__.py`, change some lines
```
### from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth_acc import GaussianRasterizationSettings, GaussianRasterizer

### rendered_image, radii, depth = rasterizer(
rendered_image, depth, acc, radii = rasterizer(

### ... and return dictionary
```


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>