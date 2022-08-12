# AlloyBERT

This repository was developed during a summer internship at the Oak Ridge National Laboratory through the Pathways to Computing Internship Program. This work is an extension on this [repository](https://github.com/peizong/alloy2vec) worked on by my mentor Dr. Junqi Yin! 

AlloyBERT is a large transformers lanaguge model trained on a dataset of more than 6 million material science publications using the Summit Supercomputer.

We used the BERT embeddings to investigate how well the model can differentiate between alloys based on their crystal structures.

Shown below is a umap plot of BERT embeddings for a subset of FCC and BCC alloys:

![umap_embeddings_of_a_subset_of alloys](https://user-images.githubusercontent.com/43448128/184356269-cc226535-c334-43cc-9427-9618698a5967.png)

Plotting the embeddings of a much larger dataset shows the model doing well in differenting between single-phase and multi-phase crystal structures:

![umap_embeddings](https://user-images.githubusercontent.com/43448128/184356980-20811037-4689-4b3d-a080-0a19ac9c4ddf.png)

We trained a simple KNN classifer on the umap embeddings of the materials in the four crystal structures and achieved a +84% classification accuracy, please check the visualizations.ipynb notebook for more details.

Finally, we tested the scaling characteristics of our model by training it on 1, 2, and 4 nodes of the summit supercomputer, as shown in the graph below, the training scalled as excepted:

![scaling_plot](https://user-images.githubusercontent.com/43448128/184357679-7f6ef97f-d22a-487a-ae10-43d4cb3d4ae2.png)


## Installation and Usage

You can easily use our trained model either for inference or you can fine tune it on your dataset. Our trained model and tokenizer can be found [here]()

### Training Instructions

1 - Clone the repository 

2 - Run the following command to install the required packages: ```pip install --ignore-installed -r requirements.txt```

3 - Run the following command to train the model on your dataset:

```
python -u multi_gpu_model_trainer.py \
      --output_dir="./output" \
      --train_data_dir="add your dataset path here" \
      --tokenizer_path="alloy-data-tokenizer"\
      --train_local_bs=64 \
      --max_epochs=1000 \
      --start_lr=1e-3 \
      --adam_eps=1e-8 \
      --weight_decay=1e-6 \
      --lr_warmup_steps=0 \
      --lr_warmup_factor=1 \
      --lr_schedule type='multistep',milestones='2500 16384',decay_rate='0.1' \
      --logging_frequency=100 \
      
```

### Inference Instructions

1 - Clone the repository 

2 - Run the following command to install the required packages: ```pip install --ignore-installed -r requirements.txt```

3 - Unzip the downloaded trained model and tokenizer in the same folder as this readme file

4 - Run through the visualization.ipynb notebook

