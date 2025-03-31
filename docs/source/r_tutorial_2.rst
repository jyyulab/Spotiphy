.. _r_tutorial_2:

Conversion of Spotiphy-Derived iscRNA Data to Seurat or SparseEset Object for Downstream Analysis
========================================================================================================================

.. code-block:: r

    library(Seurat)
    library(SeuratDisk)
    library(SeuratData)
    library(scMINER)

    ###########Create Seurat Object using Spotiphy-derived iscRNA data##############
    # Read h5ad file
    ST_decomposition <- readH5AD("ST_decomposition.h5ad")
    ST_decomposition.seurat <- as.Seurat(ST_decomposition, counts = "X", data = "X")
    ST_decomposition.seurat$samplename <- 'samplename'
    # saveRDS(ST_decomposition.seurat, file = "ST_decomposition.seurat.rds")

    #Common Seurat pipeline#
    ST_decomposition.seurat <- NormalizeData(ST_decomposition.seurat, normalization.method = "LogNormalize", scale.factor=1e6)
    ST_decomposition.seurat <- FindVariableFeatures(ST_decomposition.seurat, selection.method = "vst", nfeatures = 2000)
    ST_decomposition.seurat <- ScaleData(ST_decomposition.seurat, features = rownames(ST_decomposition.seurat))
    ST_decomposition.seurat <- RunPCA(ST_decomposition.seurat, features = VariableFeatures(object = ST_decomposition.seurat))
    ST_decomposition.seurat <- FindNeighbors(ST_decomposition.seurat, dims = 1:30)
    ST_decomposition.seurat <- FindClusters(ST_decomposition.seurat, resolution = 0.5)
    ST_decomposition.seurat <- RunUMAP(ST_decomposition.seurat, dims = 1:30)
    DimPlot(ST_decomposition.seurat, reduction = "umap")
    # saveRDS(ST_decomposition.seurat, file = "ST_decomposition.seurat.rds")

    #Find DE genes of Clusters
    markers <- FindAllMarkers(ST_decomposition.seurat, only.pos = TRUE)

    # Output cluster information for Spatial Layout in Python
    meta.data <- ST_decomposition.seurat@meta.data
    meta.data$ID <- rownames(meta.data)
    out2excel(meta.data,out.xlsx = "ST_decomposition.seurat_meta.xlsx")

    ####################Create SparseEset Object using Seurat Object################
    #build meta.data
    meta.data <- ST_decomposition.seurat@meta.data
    #create SparseEset
    ST_decomposition.eset <- createSparseEset(input_matrix=ST_decomposition.seurat@assays$originalexp@counts,cellData=meta.data)
    colnames(ST_decomposition.eset)[1:(ncol(ST_decomposition.eset))] <- sapply(1:(ncol(ST_decomposition.eset)), function(l) paste0("Cell_", colnames(ST_decomposition.eset)[l]))
    # saveRDS(ST_decomposition.eset, file = "ST_decomposition.eset.rds")

    #Common scMINER pipeline# For more details: https://jyyulab.github.io/scMINER/index.html
    #QC
    drawSparseEsetQC(input_eset = ST_decomposition.eset, output_html_file = "ST_decomposition.eset_rawCount.html", overwrite = FALSE)
    ST_decomposition.eset <- filterSparseEset(ST_decomposition.eset, filter_mode = "auto", filter_type = "both")
    #Normalization
    ST_decomposition.eset.log2 <- normalizeSparseEset(ST_decomposition.eset, scale_factor = 1000000, log_base = 2, log_pseudoCount = 1)
    # saveRDS(ST_decomposition.eset.log2, file = "ST_decomposition.eset.log2.rds")

    #Generate MICA input
    generateMICAinput(input_eset= ST_decomposition.eset.log2, output_file = "ST_decomposition.eset.log2_MICA_input.txt", overwrite = TRUE)

    #Read in MICA output for annotation
    ST_decomposition.eset.log2 <- addMICAoutput(input_eset = ST_decomposition.eset.log2, visual_method = "umap",
                                    mica_output_file = "ST_decomposition.eset.log2/clustering_UMAP_euclidean_20_2.0.txt")
    MICAplot(input_eset = ST_decomposition.eset.log2, X = "UMAP_1", Y = "UMAP_2", color_by = "clusterID", point.size = 0.5, fontsize.cluster_label = 4)
    # saveRDS(ST_decomposition.eset.log2, file = "ST_decomposition.eset.log2.rds")

    #Get DE genes of Clusters
    de_res <- getDE(input_eset = ST_decomposition.eset.log2, group_by = "clusterID", use_method = "limma")


    # Output cluster information for Spatial Layout in Python
    meta.data <- Biobase::pData(ST_decomposition.eset.log2)
    meta.data$ID <- rownames(meta.data)
    out2excel(meta.data,out.xlsx = "ST_decomposition.eset.log2_meta.xlsx")


