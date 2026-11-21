#!/usr/bin/env Rscript
# ==============================================================================
# Official MARVEL (v2.0.5, Wen et al., Nucleic Acids Research 2023) Upstream
# Single-Cell Splicing Event Quantification Adapter for Katana-PRISM.
#
# Quantifies canonical SE, MXE, RI, A5SS, and A3SS events via:
#   1. MARVEL::Preprocess_rMATS (on rMATS fromGTF.{SE,MXE,RI,A5SS,A3SS}.txt)
#   2. MARVEL::CreateMarvelObject
#   3. MARVEL::ComputePSI (SE, MXE, RI, A5SS, A3SS)
# Exports both the native `MarvelObject.rds` and Katana-PRISM's canonical 0-based
# `sc_events.tsv`, `sc_event_genes.tsv`, `sc_cells.tsv`, `sc_psi.tsv.gz`, and
# `sc_support.tsv.gz` tables (preserving MARVEL's exact PSI and read denominators).
# ==============================================================================

suppressPackageStartupMessages({
    library(methods)
    library(parallel)
})

parse_cli <- function(args) {
    res <- list(
        prep_dir = NULL,
        fromgtf_dir = NULL,
        outdir = NULL,
        marvel_tarball = NULL,
        coverage_threshold = 3,
        uneven_multiplier = 10,
        min_cells = 10,
        numerator = NULL,
        denominator = NULL,
        group_col = "condition",
        threads = 4
    )
    i <- 1
    while (i <= length(args)) {
        k <- args[i]
        v <- if (i < length(args)) args[i + 1] else NULL
        if (k == "--prep-dir") { res$prep_dir <- v; i <- i + 2 }
        else if (k == "--fromgtf-dir") { res$fromgtf_dir <- v; i <- i + 2 }
        else if (k == "--outdir") { res$outdir <- v; i <- i + 2 }
        else if (k == "--marvel-tarball") { res$marvel_tarball <- v; i <- i + 2 }
        else if (k == "--coverage-threshold") { res$coverage_threshold <- as.numeric(v); i <- i + 2 }
        else if (k == "--uneven-multiplier") { res$uneven_multiplier <- as.numeric(v); i <- i + 2 }
        else if (k == "--min-cells") { res$min_cells <- as.integer(v); i <- i + 2 }
        else if (k == "--numerator") { res$numerator <- v; i <- i + 2 }
        else if (k == "--denominator") { res$denominator <- v; i <- i + 2 }
        else if (k == "--group-col") { res$group_col <- v; i <- i + 2 }
        else if (k == "--threads") { res$threads <- as.integer(v); i <- i + 2 }
        else { i <- i + 1 }
    }
    if (is.null(res$prep_dir) || is.null(res$fromgtf_dir) || is.null(res$outdir)) {
        stop("Required arguments: --prep-dir, --fromgtf-dir, --outdir")
    }
    res
}

ensure_marvel_loaded <- function(tarball_path, outdir) {
    Sys.setenv(TAR = "internal")
    r_lib <- file.path(outdir, ".r_marvel_lib")
    dir.create(r_lib, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(r_lib, .libPaths()))

    if (requireNamespace("MARVEL", quietly = TRUE)) {
        suppressPackageStartupMessages(library(MARVEL))
        message("[MARVEL] Loaded installed MARVEL v", as.character(utils::packageVersion("MARVEL")))
        return(invisible(TRUE))
    }
    if (is.null(tarball_path) || !file.exists(tarball_path)) {
        dl_tar <- file.path(r_lib, "MARVEL_github.tar.gz")
        tryCatch({
            utils::download.file("https://codeload.github.com/wenweixiong/MARVEL/tar.gz/refs/heads/master", dl_tar, quiet = TRUE, mode = "wb")
            if (file.exists(dl_tar) && file.info(dl_tar)$size > 10000) tarball_path <- dl_tar
        }, error = function(e) NULL)
    }
    if (!is.null(tarball_path) && file.exists(tarball_path)) {
        message("[MARVEL] Installing official MARVEL source tarball into ", r_lib, " ...")
        ok <- tryCatch({
            utils::install.packages(tarball_path, lib = r_lib, repos = NULL, type = "source", quiet = TRUE)
            requireNamespace("MARVEL", quietly = TRUE)
        }, error = function(e) FALSE)
        if (isTRUE(ok)) {
            suppressPackageStartupMessages(library(MARVEL))
            message("[MARVEL] Installed & loaded MARVEL v", as.character(utils::packageVersion("MARVEL")))
            return(invisible(TRUE))
        }
        # Pure-R source fallback if ggplot2/scales graphic dependencies are absent in a minimal R image
        message("[MARVEL] Sourcing official MARVEL R/Script_PLATE_*.R functions directly ...")
        ex_dir <- file.path(r_lib, "marvel_src")
        dir.create(ex_dir, recursive = TRUE, showWarnings = FALSE)
        utils::untar(tarball_path, exdir = ex_dir, tar = "internal")
        if (requireNamespace("plyr", quietly = TRUE)) {
            assign("join", plyr::join, envir = .GlobalEnv)
        } else {
            assign("join", function(x, y, by = NULL, type = "left", ...) {
                if (is.null(by)) by <- intersect(names(x), names(y))
                x$.ord_idx__ <- seq_len(nrow(x))
                m <- merge(x, y, by = by, all.x = (type == "left"), sort = FALSE)
                m <- m[order(m$.ord_idx__), , drop = FALSE]
                m$.ord_idx__ <- NULL
                rownames(m) <- NULL
                m
            }, envir = .GlobalEnv)
        }
        r_scripts <- sort(list.files(ex_dir, pattern = "^Script_PLATE_0[01234].*\\.R$", recursive = TRUE, full.names = TRUE))
        for (rs in r_scripts) {
            source(rs, local = .GlobalEnv)
        }
        return(invisible(TRUE))
    }
    stop("MARVEL R package is not installed and no --marvel-tarball was supplied.")
}

make_marvel_tran_id <- function(df, etype) {
    names(df) <- gsub("^X", "", names(df))
    if (nrow(df) == 0) return(character(0))
    tid <- character(nrow(df))
    pos <- which(df$strand == "+")
    neg <- which(df$strand == "-")

    if (etype == "SE") {
        if (length(pos) > 0) {
            p <- df[pos, , drop = FALSE]
            tid[pos] <- paste0(p$chr, ":", p$upstreamES + 1, ":", p$upstreamEE, ":+@",
                               p$chr, ":", p$exonStart_0base + 1, ":", p$exonEnd, ":+@",
                               p$chr, ":", p$downstreamES + 1, ":", p$downstreamEE)
        }
        if (length(neg) > 0) {
            n <- df[neg, , drop = FALSE]
            tid[neg] <- paste0(n$chr, ":", n$downstreamES + 1, ":", n$downstreamEE, ":-@",
                               n$chr, ":", n$exonStart_0base + 1, ":", n$exonEnd, ":-@",
                               n$chr, ":", n$upstreamES + 1, ":", n$upstreamEE)
        }
    } else if (etype == "MXE") {
        if (length(pos) > 0) {
            p <- df[pos, , drop = FALSE]
            tid[pos] <- paste0(p$chr, ":", p$upstreamES + 1, ":", p$upstreamEE, ":+@",
                               p$chr, ":", p[["1stExonStart_0base"]] + 1, ":", p[["1stExonEnd"]], ":+@",
                               p$chr, ":", p[["2ndExonStart_0base"]] + 1, ":", p[["2ndExonEnd"]], ":+@",
                               p$chr, ":", p$downstreamES + 1, ":", p$downstreamEE)
        }
        if (length(neg) > 0) {
            n <- df[neg, , drop = FALSE]
            tid[neg] <- paste0(n$chr, ":", n$downstreamES + 1, ":", n$downstreamEE, ":-@",
                               n$chr, ":", n[["2ndExonStart_0base"]] + 1, ":", n[["2ndExonEnd"]], ":-@",
                               n$chr, ":", n[["1stExonStart_0base"]] + 1, ":", n[["1stExonEnd"]], ":-@",
                               n$chr, ":", n$upstreamES + 1, ":", n$upstreamEE)
        }
    } else if (etype == "RI") {
        if (length(pos) > 0) {
            p <- df[pos, , drop = FALSE]
            tid[pos] <- paste0(p$chr, ":", p$riExonStart_0base + 1, ":", p$upstreamEE, ":+@",
                               p$chr, ":", p$downstreamES + 1, ":", p$riExonEnd)
        }
        if (length(neg) > 0) {
            n <- df[neg, , drop = FALSE]
            tid[neg] <- paste0(n$chr, ":", n$riExonEnd, ":", n$downstreamES + 1, ":-@",
                               n$chr, ":", n$upstreamEE, ":", n$riExonStart_0base + 1)
        }
    } else if (etype == "A5SS") {
        if (length(pos) > 0) {
            p <- df[pos, , drop = FALSE]
            tid[pos] <- paste0(p$chr, ":", p$longExonStart_0base + 1, ":", p$shortEE, "|", p$longExonEnd, ":+@",
                               p$chr, ":", p$flankingES + 1, ":", p$flankingEE)
        }
        if (length(neg) > 0) {
            n <- df[neg, , drop = FALSE]
            tid[neg] <- paste0(n$chr, ":", n$longExonEnd, ":", n$longExonStart_0base + 1, "|", n$shortES + 1, ":-@",
                               n$chr, ":", n$flankingES + 1, ":", n$flankingEE)
        }
    } else if (etype == "A3SS") {
        if (length(pos) > 0) {
            p <- df[pos, , drop = FALSE]
            tid[pos] <- paste0(p$chr, ":", p$flankingES + 1, ":", p$flankingEE, ":+@",
                               p$chr, ":", p$longExonStart_0base + 1, "|", p$shortES + 1, ":", p$longExonEnd)
        }
        if (length(neg) > 0) {
            n <- df[neg, , drop = FALSE]
            tid[neg] <- paste0(n$chr, ":", n$flankingES + 1, ":", n$flankingEE, ":-@",
                               n$chr, ":", n$shortEE, "|", n$longExonEnd, ":", n$longExonStart_0base + 1)
        }
    }
    tid
}

canonicalize_rmats_coords <- function(df, etype, tran_ids) {
    names(df) <- gsub("^X", "", names(df))
    n <- nrow(df)
    na_i <- rep(NA_integer_, n)
    get_c <- function(col) if (col %in% names(df)) as.integer(df[[col]]) else na_i
    gid <- gsub('"', '', as.character(df$GeneID))
    gsym <- gsub('"', '', as.character(df$geneSymbol))
    gsym[is.na(gsym) | gsym == "" | gsym == "NA"] <- sub("\\..*$", "", gid[is.na(gsym) | gsym == "" | gsym == "NA"])

    out <- data.frame(
        tran_id = tran_ids,
        event_id = paste0(etype, ":", tran_ids),
        event_type = rep(etype, n),
        gene_id = gid,
        gene_symbol = gsym,
        chr = as.character(df$chr),
        strand = as.character(df$strand),
        exonStart_0base = get_c("exonStart_0base"),
        exonEnd = get_c("exonEnd"),
        upstreamES = get_c("upstreamES"),
        upstreamEE = get_c("upstreamEE"),
        downstreamES = get_c("downstreamES"),
        downstreamEE = get_c("downstreamEE"),
        `1stExonStart_0base` = get_c("1stExonStart_0base"),
        `1stExonEnd` = get_c("1stExonEnd"),
        `2ndExonStart_0base` = get_c("2ndExonStart_0base"),
        `2ndExonEnd` = get_c("2ndExonEnd"),
        riExonStart_0base = get_c("riExonStart_0base"),
        riExonEnd = get_c("riExonEnd"),
        longExonStart_0base = get_c("longExonStart_0base"),
        longExonEnd = get_c("longExonEnd"),
        shortES = get_c("shortES"),
        shortEE = get_c("shortEE"),
        flankingES = get_c("flankingES"),
        flankingEE = get_c("flankingEE"),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
    out[!duplicated(out$tran_id), , drop = FALSE]
}

main <- function() {
    opt <- parse_cli(commandArgs(trailingOnly = TRUE))
    dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
    ensure_marvel_loaded(opt$marvel_tarball, opt$outdir)
    if (requireNamespace("plyr", quietly = TRUE)) {
        suppressPackageStartupMessages(library(plyr))
    }

    message("[MARVEL 1/4] Loading GTF, metadata, and SpliceJunction counts ...")
    gtf_df <- utils::read.table(
        file.path(opt$prep_dir, "marvel_host.gtf"),
        sep = "\t", header = FALSE, quote = "", comment.char = "#", stringsAsFactors = FALSE
    )
    # Assign global `gtf` explicitly because MARVEL v2.0.5 Preprocess_rMATS references lowercase `gtf`
    assign("gtf", gtf_df, envir = .GlobalEnv)

    pheno_orig <- utils::read.delim(file.path(opt$prep_dir, "marvel_pheno.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
    orig_ids <- as.character(pheno_orig[["sample.id"]])
    safe_ids <- sprintf("CELL_%06d", seq_along(orig_ids))
    safe_to_orig <- stats::setNames(orig_ids, safe_ids)
    orig_to_safe <- stats::setNames(safe_ids, orig_ids)

    pheno_m <- pheno_orig
    pheno_m[["sample.id"]] <- safe_ids

    gf_df <- utils::read.delim(file.path(opt$prep_dir, "marvel_gene_feature.tsv"), stringsAsFactors = FALSE, check.names = FALSE)
    sj_df <- utils::read.delim(gzfile(file.path(opt$prep_dir, "marvel_sj_counts.tsv.gz")), stringsAsFactors = FALSE, check.names = FALSE)
    sj_cell_cols <- intersect(orig_ids, names(sj_df)[-1])
    sj_df <- sj_df[, c("coord.intron", sj_cell_cols), drop = FALSE]
    names(sj_df)[-1] <- orig_to_safe[sj_cell_cols]

    ri_file <- file.path(opt$prep_dir, "marvel_ri_intron_counts.tsv.gz")
    has_ri <- file.exists(ri_file)
    ri_df <- NULL
    if (has_ri) {
        ri_df <- utils::read.delim(gzfile(ri_file), stringsAsFactors = FALSE, check.names = FALSE)
        ri_df <- ri_df[, c("coord.intron", sj_cell_cols), drop = FALSE]
        names(ri_df)[-1] <- orig_to_safe[sj_cell_cols]
    }
    pheno_m <- pheno_m[match(names(sj_df)[-1], pheno_m[["sample.id"]]), , drop = FALSE]

    message("[MARVEL 2/4] Running MARVEL::Preprocess_rMATS across SE, MXE, RI, A5SS, A3SS ...")
    etypes <- c("SE", "MXE", "RI", "A5SS", "A3SS")
    splice_feature <- list()
    coord_lookup <- list()
    for (et in etypes) {
        f_p <- file.path(opt$fromgtf_dir, paste0("fromGTF.", et, ".txt"))
        if (!file.exists(f_p)) next
        raw_df <- utils::read.table(f_p, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")
        if (nrow(raw_df) == 0) next
        tids <- make_marvel_tran_id(raw_df, et)
        coord_lookup[[et]] <- canonicalize_rmats_coords(raw_df, et, tids)
        splice_feature[[et]] <- Preprocess_rMATS(file = raw_df, GTF = gtf_df, EventType = et)
        message(sprintf("  [%s] %d annotated events preprocessed", et, nrow(splice_feature[[et]])))
    }

    message("[MARVEL 3/4] Building MarvelObject & running MARVEL::ComputePSI ...")
    marvel <- CreateMarvelObject(
        SplicePheno = pheno_m,
        SpliceJunction = sj_df,
        IntronCounts = ri_df,
        SpliceFeature = splice_feature,
        GeneFeature = gf_df,
        GTF = gtf_df
    )

    for (et in c("SE", "MXE", "A5SS", "A3SS")) {
        if (!is.null(marvel$SpliceFeature[[et]]) && nrow(marvel$SpliceFeature[[et]]) > 0) {
            message("  -> MARVEL::ComputePSI EventType = ", et)
            marvel <- tryCatch(
                ComputePSI(
                    MarvelObject = marvel,
                    CoverageThreshold = opt$coverage_threshold,
                    UnevenCoverageMultiplier = opt$uneven_multiplier,
                    EventType = et
                ),
                error = function(e) {
                    message("  [WARN] ComputePSI(", et, ") skipped: ", conditionMessage(e))
                    marvel
                }
            )
        }
    }
    if (has_ri && !is.null(marvel$SpliceFeature[["RI"]]) && nrow(marvel$SpliceFeature[["RI"]]) > 0) {
        ri_threads <- min(4L, max(2L, opt$threads))
        message("  -> MARVEL::ComputePSI EventType = RI (threads = ", ri_threads, ")")
        marvel <- tryCatch(
            ComputePSI(
                MarvelObject = marvel,
                CoverageThreshold = opt$coverage_threshold,
                IntronCounts = marvel$IntronCounts,
                thread = ri_threads,
                read.length = 1,
                EventType = "RI"
            ),
            error = function(e) {
                message("  [WARN] ComputePSI(RI) skipped: ", conditionMessage(e))
                marvel
            }
        )
    }

    rds_path <- file.path(opt$outdir, "MarvelObject.rds")
    saveRDS(marvel, rds_path)
    message("[MARVEL] Saved native MarvelObject -> ", rds_path)

    message("[MARVEL 4/4] Exporting canonical 0-based event & support matrices for Katana-PRISM ...")
    safe_cells <- pheno_m[["sample.id"]]
    out_cells <- unname(safe_to_orig[safe_cells])

    ev_blocks <- list()
    psi_blocks <- list()
    sup_blocks <- list()

    for (et in etypes) {
        psi_df <- marvel$PSI[[et]]
        if (is.null(psi_df) || nrow(psi_df) == 0) next
        tids <- as.character(psi_df$tran_id)
        p_mat <- as.matrix(psi_df[, safe_cells, drop = FALSE])
        storage.mode(p_mat) <- "double"

        # Extract exact competing read denominators from MARVEL$Counts[[et]]
        cnt <- marvel$Counts[[et]]
        if (et == "SE") {
            i1 <- as.matrix(cnt$sj.included.1[, safe_cells, drop = FALSE])
            i2 <- as.matrix(cnt$sj.included.2[, safe_cells, drop = FALSE])
            ex <- as.matrix(cnt$sj.excluded[, safe_cells, drop = FALSE])
            s_mat <- 0.5 * (i1 + i2) + ex
        } else if (et == "MXE") {
            i1 <- as.matrix(cnt$sj.included.1[, safe_cells, drop = FALSE])
            i2 <- as.matrix(cnt$sj.included.2[, safe_cells, drop = FALSE])
            e1 <- as.matrix(cnt$sj.excluded.1[, safe_cells, drop = FALSE])
            e2 <- as.matrix(cnt$sj.excluded.2[, safe_cells, drop = FALSE])
            s_mat <- 0.5 * (i1 + i2 + e1 + e2)
        } else if (et %in% c("A5SS", "A3SS")) {
            inc <- as.matrix(cnt$sj.included[, safe_cells, drop = FALSE])
            exc <- as.matrix(cnt$sj.excluded[, safe_cells, drop = FALSE])
            s_mat <- inc + exc
        } else if (et == "RI") {
            inc <- as.matrix(cnt$counts.included[, safe_cells, drop = FALSE])
            exc <- as.matrix(cnt$counts.excluded[, safe_cells, drop = FALSE])
            s_mat <- inc + exc
        }
        storage.mode(s_mat) <- "double"
        s_mat[is.na(p_mat)] <- 0.0

        cl <- coord_lookup[[et]]
        idx_m <- match(tids, cl$tran_id)
        valid_m <- !is.na(idx_m)
        if (!any(valid_m)) next

        meta_et <- cl[idx_m[valid_m], , drop = FALSE]
        p_mat <- p_mat[valid_m, , drop = FALSE]
        s_mat <- s_mat[valid_m, , drop = FALSE]

        n_obs <- rowSums(is.finite(p_mat))
        m_psi <- rowMeans(p_mat, na.rm = TRUE)
        min_c <- min(opt$min_cells, max(2L, ncol(p_mat) %/% 4L))
        keep <- (n_obs >= min_c) & is.finite(m_psi) & (m_psi >= 0.01) & (m_psi <= 0.99)
        if (!any(keep)) next

        ev_blocks[[et]] <- meta_et[keep, , drop = FALSE]
        psi_blocks[[et]] <- p_mat[keep, , drop = FALSE]
        sup_blocks[[et]] <- s_mat[keep, , drop = FALSE]
        message(sprintf("  [%s] %d active events exported (observed in >= %d cells)", et, sum(keep), min_c))
    }

    if (length(ev_blocks) == 0) {
        stop("Zero splicing events passed MARVEL quantification and cell-observation thresholds.")
    }

    ev_all <- do.call(rbind, ev_blocks)
    rownames(ev_all) <- NULL
    psi_all <- do.call(rbind, psi_blocks)
    sup_all <- do.call(rbind, sup_blocks)

    # Standardize cell metadata for Katana-PRISM downstream modules
    md_out <- pheno_orig[match(out_cells, orig_ids), , drop = FALSE]
    md_out[["sample.id"]] <- NULL
    if (!("cell_id" %in% names(md_out))) {
        cb_col <- intersect(c("cell_barcode", "barcode", "sample_id"), names(md_out))
        md_out$cell_id <- if (length(cb_col) > 0) as.character(md_out[[cb_col[1]]]) else out_cells
    }
    if (!("donor_id" %in% names(md_out))) {
        dc <- intersect(c("mouse_id", "donor", "biological_sample_id", "subject"), names(md_out))
        md_out$donor_id <- if (length(dc) > 0) as.character(md_out[[dc[1]]]) else "Donor1"
    }
    if (!("condition" %in% names(md_out)) && opt$group_col %in% names(md_out)) {
        md_out$condition <- as.character(md_out[[opt$group_col]])
    }
    if (!("qc_pass" %in% names(md_out))) md_out$qc_pass <- TRUE

    # Preliminary two-group dPSI/FDR summary column (strictly BH-adjusted across testable events only)
    dpsi_v <- rep(NA_real_, nrow(ev_all))
    pval_v <- rep(NA_real_, nrow(ev_all))
    cmp_id <- "Cohort_All"
    if (!is.null(opt$numerator) && !is.null(opt$denominator) && ("condition" %in% names(md_out))) {
        cmp_id <- paste0(opt$numerator, "_vs_", opt$denominator)
        g1 <- which(as.character(md_out$condition) == opt$numerator)
        g2 <- which(as.character(md_out$condition) == opt$denominator)
        if (length(g1) >= 2 && length(g2) >= 2) {
            for (i in seq_len(nrow(ev_all))) {
                v1 <- psi_all[i, g1]; v1 <- v1[is.finite(v1)]
                v2 <- psi_all[i, g2]; v2 <- v2[is.finite(v2)]
                if (length(v1) >= 3 && length(v2) >= 3) {
                    dpsi_v[i] <- mean(v1) - mean(v2)
                    if (stats::sd(c(v1, v2)) > 0) {
                        pval_v[i] <- suppressWarnings(stats::wilcox.test(v1, v2)$p.value)
                    }
                }
            }
        }
    }
    fdr_v <- rep(NA_real_, nrow(ev_all))
    ok_p <- is.finite(pval_v)
    if (any(ok_p)) {
        fdr_v[ok_p] <- stats::p.adjust(pval_v[ok_p], method = "BH")
    }
    ev_all$dpsi <- round(dpsi_v, 5)
    ev_all$pvalue <- pval_v
    ev_all$fdr <- fdr_v
    ev_all$comparison_id <- cmp_id
    ev_all$tran_id <- NULL

    utils::write.table(ev_all, file.path(opt$outdir, "sc_events.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, na = "")
    utils::write.table(ev_all[, c("event_id", "gene_id", "gene_symbol")], file.path(opt$outdir, "sc_event_genes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
    utils::write.table(md_out, file.path(opt$outdir, "sc_cells.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

    psi_t <- as.data.frame(t(round(psi_all, 5)), check.names = FALSE)
    colnames(psi_t) <- ev_all$event_id
    psi_t <- cbind(data.frame(cell_id = out_cells, stringsAsFactors = FALSE), psi_t)
    gz_psi <- gzfile(file.path(opt$outdir, "sc_psi.tsv.gz"), "wt")
    utils::write.table(psi_t, gz_psi, sep = "\t", quote = FALSE, row.names = FALSE)
    close(gz_psi)

    sup_t <- as.data.frame(t(round(sup_all, 2)), check.names = FALSE)
    colnames(sup_t) <- ev_all$event_id
    sup_t <- cbind(data.frame(cell_id = out_cells, stringsAsFactors = FALSE), sup_t)
    gz_sup <- gzfile(file.path(opt$outdir, "sc_support.tsv.gz"), "wt")
    utils::write.table(sup_t, gz_sup, sep = "\t", quote = FALSE, row.names = FALSE)
    close(gz_sup)

    message(sprintf("[MARVEL COMPLETE] Quantified %d total events across %d cells -> %s", nrow(ev_all), length(out_cells), opt$outdir))
}

main()
