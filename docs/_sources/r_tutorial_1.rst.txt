.. _r_tutorial_1:

Single-cell RNA-seq Data Input Preparation from a Seurat Object
====================

.. code-block:: r

    library(Seurat)
    library(SeuratDisk)
    library(SeuratData)

    #Common Seurat pipeline#
    SeuratSample <- NormalizeData(SeuratSample, normalization.method = "LogNormalize", scale.factor=1e6)
    SeuratSample <- FindVariableFeatures(SeuratSample, selection.method = "vst", nfeatures = 2000)
    SeuratSample <- ScaleData(SeuratSample, features = rownames(SeuratSample))
    SeuratSample <- RunPCA(SeuratSample, features = VariableFeatures(object = SeuratSample))
    SeuratSample <- FindNeighbors(SeuratSample, dims = 1:30)
    SeuratSample <- FindClusters(SeuratSample, resolution = 0.5)
    SeuratSample <- RunUMAP(SeuratSample, dims = 1:30)
    DimPlot(SeuratSample, reduction = "umap")
    # saveRDS(SeuratSample, file = "SeuratSample.rds")

    ###Convert Seurat Object to h5ad file as input for deconvolution in Python###
    #Make sure raw counts will be used for conversion
    SeuratSample@assays$RNA@data = SeuratSample@assays$RNA@counts
    SeuratSample@assays$RNA@scale.data = as.matrix(SeuratSample@assays$RNA@counts)
    SaveH5Seurat(SeuratSample, filename = "SeuratSample.h5Seurat")
    Convert("SeuratSample.h5Seurat", dest = "h5ad")


