# aivivn-vn-tones

## Generate data
Generate data for default problem translate_vndt
```bash
./gen_data.sh
```

Generate data for custom problem A
```bash
./gen_data.sh A
```

## Train model
On problem `translate_vndt`, to train model `transformer` with hparams `transformer_base` on GPUs `0,1`
```bash
./train.sh 0,1 transformer_base transformer translate_vndt
```

## Predict
Similar to `train.sh`
```bash
./predict.sh 0,1 transformer_base transformer translate_vndt
```

The output is stored in `sub-translate_vndt-transformer-transformer_base.csv`