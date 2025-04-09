to reproduce the result, simply run:

```py
python KMVE_RG/test_metric.py --model AllOrgan --ckpt Result/TF_organ_balence_sampler-4090/Models/all_best.pth --method balsam --comment balsam
```


result is as followed:

```csv
Mammary,balsam,0.769,0.719,0.681,0.650,0.477,0.772,0,0,0,0
Liver,balsam,0.893,0.866,0.845,0.827,0.562,0.880,0,0,0,0
Thyroid,balsam,0.757,0.696,0.644,0.599,0.460,0.755,0,0,0,0
```
