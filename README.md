# text_gcn
Pytorch implementation to paper "Graph Convolutional Networks for Text Classification".
## Running
You can run the whole process very easily. Take the R8 corpus for example,

### Step 1: Clean the corpus.
```bash
./scripts/R8.sh clean
```

### Step 2: Build dataset.
```bash
./scripts/R8.sh build_dataset
```

### Step 3: Build graph.
```bash
./scripts/R8.sh build_graph
```

### Step 4: Train and inference.
```bash
./scripts/R8.sh train
```
## Performance Comparision

-|Dataset|Accuracy
:-:|:-:|:-:
Original|R8|97.07%
This Implementation|R8|96.21%

-|Dataset|Accuracy
:-:|:-:|:-:
Original|R52|93.56%
This Implementation|R52|92.52%
