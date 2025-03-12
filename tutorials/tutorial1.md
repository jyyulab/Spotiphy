

``` r
library(Seurat)
```

```
## Error in library(Seurat): there is no package called 'Seurat'
```

``` r
library(SeuratDisk)
```

```
## Error in library(SeuratDisk): there is no package called 'SeuratDisk'
```

``` r
library(SeuratData)
```

```
## Error in library(SeuratData): there is no package called 'SeuratData'
```

``` r
#Common Seurat pipeline#
SeuratSample <- NormalizeData(SeuratSample, normalization.method = "LogNormalize", scale.factor=1e6)
```

```
## Error in NormalizeData(SeuratSample, normalization.method = "LogNormalize", : could not find function "NormalizeData"
```

``` r
SeuratSample <- FindVariableFeatures(SeuratSample, selection.method = "vst", nfeatures = 2000)
```

```
## Error in FindVariableFeatures(SeuratSample, selection.method = "vst", : could not find function "FindVariableFeatures"
```

``` r
SeuratSample <- ScaleData(SeuratSample, features = rownames(SeuratSample))
```

```
## Error in ScaleData(SeuratSample, features = rownames(SeuratSample)): could not find function "ScaleData"
```

``` r
SeuratSample <- RunPCA(SeuratSample, features = VariableFeatures(object = SeuratSample))
```

```
## Error in RunPCA(SeuratSample, features = VariableFeatures(object = SeuratSample)): could not find function "RunPCA"
```

``` r
SeuratSample <- FindNeighbors(SeuratSample, dims = 1:30)
```

```
## Error in FindNeighbors(SeuratSample, dims = 1:30): could not find function "FindNeighbors"
```

``` r
SeuratSample <- FindClusters(SeuratSample, resolution = 0.5)
```

```
## Error in FindClusters(SeuratSample, resolution = 0.5): could not find function "FindClusters"
```

``` r
SeuratSample <- RunUMAP(SeuratSample, dims = 1:30)
```

```
## Error in RunUMAP(SeuratSample, dims = 1:30): could not find function "RunUMAP"
```

``` r
DimPlot(SeuratSample, reduction = "umap")
```

```
## Error in DimPlot(SeuratSample, reduction = "umap"): could not find function "DimPlot"
```

``` r
# saveRDS(SeuratSample, file = "SeuratSample.rds")

###Convert Seurat Object to h5ad file as input for deconvolution in Python###
#Make sure raw counts will be used for conversion
SeuratSample@assays$RNA@data = SeuratSample@assays$RNA@counts
```

```
## Error in eval(expr, envir, enclos): object 'SeuratSample' not found
```

``` r
SeuratSample@assays$RNA@scale.data = as.matrix(SeuratSample@assays$RNA@counts)
```

```
## Error in eval(expr, envir, enclos): object 'SeuratSample' not found
```

``` r
SaveH5Seurat(SeuratSample, filename = "SeuratSample.h5Seurat")
```

```
## Error in SaveH5Seurat(SeuratSample, filename = "SeuratSample.h5Seurat"): could not find function "SaveH5Seurat"
```

``` r
Convert("SeuratSample.h5Seurat", dest = "h5ad")
```

```
## Error in Convert("SeuratSample.h5Seurat", dest = "h5ad"): could not find function "Convert"
```

