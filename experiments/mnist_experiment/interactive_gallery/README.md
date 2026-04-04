# Interactive MNIST Sphere

This folder contains a standalone static Plotly viewer for the qualitative MNIST `spcauchy` `S^2` run.

## Source run

- `experiments/mnist_experiment/outputs/qualitative/spcauchy_s2/seed_1`
- checkpoint: `best_recon_checkpoint.pt`

## Build

Run:

```powershell
python experiments\mnist_experiment\interactive_gallery\exporters\build_site.py
```

This writes:

- `experiments/mnist_experiment/interactive_gallery/site/data/mnist_spcauchy_s2.json`

If the qualitative checkpoint is available locally, the payload is rebuilt from the saved model and MNIST eval split.
If the checkpoint is not available, the builder reuses the already committed JSON payload so GitHub Pages deployment can still succeed.

## Local preview

Serve the site with a local static server:

```powershell
python -m http.server 8000 --directory experiments\mnist_experiment\interactive_gallery\site
```

Then open:

```text
http://127.0.0.1:8000/
```

## Deployment

The GitHub Pages workflow lives in:

- `.github/workflows/deploy-mnist-gallery.yml`

It deploys:

- `experiments/mnist_experiment/interactive_gallery/site`
