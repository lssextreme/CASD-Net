# Pretrained backbone weights

Place the SAM ViT-B checkpoint here:

```
weights/sam_vit_b_01ec64.pth
```

Download it from Meta AI's Segment Anything release
(`sam_vit_b_01ec64.pth`, ~375 MB) and move it into this folder. The file is not
bundled with this repository.

At build time, `casdnet.CASDNet` loads this checkpoint to initialise the frozen
encoder; if the file is missing, `MedSAM/models/sam/build_sam.py` will offer to
download it interactively.
