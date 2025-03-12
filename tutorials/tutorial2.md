

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
library(scMINER)
```

```
## Error in library(scMINER): there is no package called 'scMINER'
```

``` r
###########Create Seurat Object using Spotiphy-derived iscRNA data##############
# Read h5ad file
ST_decomposition <- readH5AD("ST_decomposition.h5ad")
```

```
## Error in readH5AD("ST_decomposition.h5ad"): could not find function "readH5AD"
```

``` r
ST_decomposition.seurat <- as.Seurat(ST_decomposition, counts = "X", data = "X")
```

```
## Error in as.Seurat(ST_decomposition, counts = "X", data = "X"): could not find function "as.Seurat"
```

``` r
ST_decomposition.seurat$samplename <- 'samplename'
```

```
## Error: object 'ST_decomposition.seurat' not found
```

``` r
# saveRDS(ST_decomposition.seurat, file = "ST_decomposition.seurat.rds")

#Common Seurat pipeline#
ST_decomposition.seurat <- NormalizeData(ST_decomposition.seurat, normalization.method = "LogNormalize", scale.factor=1e6)
```

```
## Error in NormalizeData(ST_decomposition.seurat, normalization.method = "LogNormalize", : could not find function "NormalizeData"
```

``` r
ST_decomposition.seurat <- FindVariableFeatures(ST_decomposition.seurat, selection.method = "vst", nfeatures = 2000)
```

```
## Error in FindVariableFeatures(ST_decomposition.seurat, selection.method = "vst", : could not find function "FindVariableFeatures"
```

``` r
ST_decomposition.seurat <- ScaleData(ST_decomposition.seurat, features = rownames(ST_decomposition.seurat))
```

```
## Error in ScaleData(ST_decomposition.seurat, features = rownames(ST_decomposition.seurat)): could not find function "ScaleData"
```

``` r
ST_decomposition.seurat <- RunPCA(ST_decomposition.seurat, features = VariableFeatures(object = ST_decomposition.seurat))
```

```
## Error in RunPCA(ST_decomposition.seurat, features = VariableFeatures(object = ST_decomposition.seurat)): could not find function "RunPCA"
```

``` r
ST_decomposition.seurat <- FindNeighbors(ST_decomposition.seurat, dims = 1:30)  
```

```
## Error in FindNeighbors(ST_decomposition.seurat, dims = 1:30): could not find function "FindNeighbors"
```

``` r
ST_decomposition.seurat <- FindClusters(ST_decomposition.seurat, resolution = 0.5)
```

```
## Error in FindClusters(ST_decomposition.seurat, resolution = 0.5): could not find function "FindClusters"
```

``` r
ST_decomposition.seurat <- RunUMAP(ST_decomposition.seurat, dims = 1:30)   
```

```
## Error in RunUMAP(ST_decomposition.seurat, dims = 1:30): could not find function "RunUMAP"
```

``` r
DimPlot(ST_decomposition.seurat, reduction = "umap")
```

```
## Error in DimPlot(ST_decomposition.seurat, reduction = "umap"): could not find function "DimPlot"
```

``` r
# saveRDS(ST_decomposition.seurat, file = "ST_decomposition.seurat.rds")

#Find DE genes of Clusters
markers <- FindAllMarkers(ST_decomposition.seurat, only.pos = TRUE)
```

```
## Error in FindAllMarkers(ST_decomposition.seurat, only.pos = TRUE): could not find function "FindAllMarkers"
```

``` r
# Output cluster information for Spatial Layout in Python
meta.data <- ST_decomposition.seurat@meta.data
```

```
## Error in eval(expr, envir, enclos): object 'ST_decomposition.seurat' not found
```

``` r
meta.data$ID <- rownames(meta.data)
```

```
## Error in eval(expr, envir, enclos): object 'meta.data' not found
```

``` r
out2excel(meta.data,out.xlsx = "ST_decomposition.seurat_meta.xlsx")
```

```
## Error in out2excel(meta.data, out.xlsx = "ST_decomposition.seurat_meta.xlsx"): could not find function "out2excel"
```

``` r
################################################################################







####################Create SparseEset Object using Seurat Object################
#build meta.data
meta.data <- ST_decomposition.seurat@meta.data
```

```
## Error in eval(expr, envir, enclos): object 'ST_decomposition.seurat' not found
```

``` r
#create SparseEset
ST_decomposition.eset <- createSparseEset(input_matrix=ST_decomposition.seurat@assays$originalexp@counts,cellData=meta.data)
```

```
## Error in createSparseEset(input_matrix = ST_decomposition.seurat@assays$originalexp@counts, : could not find function "createSparseEset"
```

``` r
colnames(ST_decomposition.eset)[1:(ncol(ST_decomposition.eset))] <- sapply(1:(ncol(ST_decomposition.eset)), function(l) paste0("Cell_", colnames(ST_decomposition.eset)[l]))
```

```
## Error in eval(expr, envir, enclos): object 'ST_decomposition.eset' not found
```

``` r
# saveRDS(ST_decomposition.eset, file = "ST_decomposition.eset.rds")

#Common scMINER pipeline# For more details: https://jyyulab.github.io/scMINER/index.html
#QC
drawSparseEsetQC(input_eset = ST_decomposition.eset, output_html_file = "ST_decomposition.eset_rawCount.html", overwrite = FALSE)
```

```
## Error in drawSparseEsetQC(input_eset = ST_decomposition.eset, output_html_file = "ST_decomposition.eset_rawCount.html", : could not find function "drawSparseEsetQC"
```

``` r
ST_decomposition.eset <- filterSparseEset(ST_decomposition.eset, filter_mode = "auto", filter_type = "both")
```

```
## Error in filterSparseEset(ST_decomposition.eset, filter_mode = "auto", : could not find function "filterSparseEset"
```

``` r
#Normalization
ST_decomposition.eset.log2 <- normalizeSparseEset(ST_decomposition.eset, scale_factor = 1000000, log_base = 2, log_pseudoCount = 1)
```

```
## Error in normalizeSparseEset(ST_decomposition.eset, scale_factor = 1e+06, : could not find function "normalizeSparseEset"
```

``` r
# saveRDS(ST_decomposition.eset.log2, file = "ST_decomposition.eset.log2.rds")

#Generate MICA input
generateMICAinput(input_eset= ST_decomposition.eset.log2, output_file = "ST_decomposition.eset.log2_MICA_input.txt", overwrite = TRUE)
```

```
## Error in generateMICAinput(input_eset = ST_decomposition.eset.log2, output_file = "ST_decomposition.eset.log2_MICA_input.txt", : could not find function "generateMICAinput"
```

``` r
#Read in MICA output for annotation
ST_decomposition.eset.log2 <- addMICAoutput(input_eset = ST_decomposition.eset.log2, visual_method = "umap", 
                                mica_output_file = "ST_decomposition.eset.log2/clustering_UMAP_euclidean_20_2.0.txt")
```

```
## Error in addMICAoutput(input_eset = ST_decomposition.eset.log2, visual_method = "umap", : could not find function "addMICAoutput"
```

``` r
MICAplot(input_eset = ST_decomposition.eset.log2, X = "UMAP_1", Y = "UMAP_2", color_by = "clusterID", point.size = 0.5, fontsize.cluster_label = 4)
```

```
## Error in MICAplot(input_eset = ST_decomposition.eset.log2, X = "UMAP_1", : could not find function "MICAplot"
```

``` r
# saveRDS(ST_decomposition.eset.log2, file = "ST_decomposition.eset.log2.rds")

#Get DE genes of Clusters
de_res <- getDE(input_eset = ST_decomposition.eset.log2, group_by = "clusterID", use_method = "limma")
```

```
## Error in getDE(input_eset = ST_decomposition.eset.log2, group_by = "clusterID", : could not find function "getDE"
```

``` r
# Output cluster information for Spatial Layout in Python
meta.data <- Biobase::pData(ST_decomposition.eset.log2)
```

```
## Error in loadNamespace(x): there is no package called 'Biobase'
```

``` r
meta.data$ID <- rownames(meta.data)
```

```
## Error in eval(expr, envir, enclos): object 'meta.data' not found
```

``` r
out2excel(meta.data,out.xlsx = "ST_decomposition.eset.log2_meta.xlsx")
```

```
## Error in out2excel(meta.data, out.xlsx = "ST_decomposition.eset.log2_meta.xlsx"): could not find function "out2excel"
```

