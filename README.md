# GCN
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

### Step 4: Train.
```bash
./scripts/ontonotes.sh train
```
