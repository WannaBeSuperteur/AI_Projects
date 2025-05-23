
* [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset from [Official PyTorch Implementation](https://github.com/ivezakis/effisegnet/tree/main)
* Segmented Polyp Dataset for Computer Aided Gastrointestinal Disease Detection

## Citation

```
@inproceedings{jha2020kvasir,
               title={Kvasir-seg: A segmented polyp dataset},
               author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Johansen, H{\aa}vard D},
               booktitle={International Conference on Multimedia Modeling},
               pages={451--462},
               year={2020},
               organization={Springer}
}
```

## How to prepare dataset

* 1. Download dataset from [here](https://github.com/ivezakis/effisegnet/tree/main/Kvasir-SEG).
* 2. Copy & Paste images and mask files as the structure below:

```
2025_05_22_Improve_EffiSegNet
- Kvasir-SEG
  - train
    - images
      - 0.jpg
      - ...
    - masks
      - 0.jpg
      - ...
  - validation
    - images
      - 0.jpg
      - ...
    - masks
      - 0.jpg
      - ...
  - test
    - images
      - 0.jpg
      - ...
    - masks
      - 0.jpg
      - ...
- ...
```