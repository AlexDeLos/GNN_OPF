<p align="center">
  <img src="https://d2k0ddhflgrk1i.cloudfront.net/Websections/Huisstijl/Bouwstenen/Logo/02-Visual-Bouwstenen-Logo-Varianten-v1.png"/><br>
  <br><br>
</p>

## Authors

Francesco Piccoli

Àlex De Los Santos Subirats

Marcus Plesner

Marco van Veen

# Inductive GNNs for power grids
In this project we aim to discover a method to aid human operators in the automatic monitoring of a constantly changing power network using Graph Neural Networks (GNNs). Additionally, we aim to provide a baseline on how well existing inductive GNNs work with problems in the power systems field that require extensive calculations to solve.

Exisiting GNNs approaches used to solve power system problems are transductive, and require to be re-trained in order to function whenever the networks are modified. We plan on using the Optimal Power Flow (OPF) problem as a baseline to answer our main research question: **“How well do inductive GNNs work on graphs that model a power system?”. The goal is to find a GNN architecture that has good inductive performance on a power system problem.**


## Codebase setup

```
**macOS/Linux**
git clone https://github.com/AlexDeLos/GNN_OPF.git
cd GNN_OPF
virtualenv gnn_opf_env -p python3.10.8 
source gnn_opf_env/bin/activate
pip3 install -r requirements.txt
```

*Installing pytorch geometric dependencies can take around 1 hour*

## Codebase Structure

## Results
