# UKBB Panel C — chi-squared significance maps (multi-fold IG aggregation).
#
# Plots −log₁₀ FDR-adjusted p-value per voxel for the H₀ "voxel is not
# relevant", aggregated across n=100 subjects × 5 folds via:
#   1. Per (subject, fold): IG attribution, smoothed σ=1.
#   2. Per subject: average across 5 folds.
#   3. Per subject: σ̂ = 1.4826 · MAD(r̄ over brain mask).
#   4. Per voxel: T_v = Σᵢ r̄²_{i,v} / σ̂_i² ~ χ²_n,  n=100.
#   5. p_v = 1 − F_{χ²_n}(T_v), Benjamini–Hochberg FDR adjusted across brain.
#
# Columns (matching main-figure terminology):
#   No Control (Balanced training)    — base_full @ coef=0
#   No Control (Confounded training)  — base_full @ coef=2
#   Age Control (Confounded training) — posthoc_age @ coef=2
# Rows: sagittal, coronal, axial.
#
# Threshold: −log₁₀ p_fdr ≥ 1.30 (i.e. p_fdr ≤ 0.05) shown; below that,
# transparent so the gray template brain shows through.
#
# Run:
#   Rscript experiments/ukbb/lrp/UKBB_lrp_panel.R

library(ggplot2)
library(grid)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR  <- "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2/"
CHI2_DIR <- paste0(RUN_DIR, "chi2/")
TPL_DIR  <- paste0(RUN_DIR, "lrp_maps/")
OUT_DIR  <- paste0(RUN_DIR, "graphics/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_PATH_MAIN     <- paste0(OUT_DIR, "Fig_UKBB_lrp_main.pdf")
OUT_PATH_APPENDIX <- paste0(OUT_DIR, "Fig_UKBB_lrp_appendix.pdf")

# ---------------------------------------------------------------------------
# Colours + scale
# ---------------------------------------------------------------------------
OI_VERMILLION <- "#D55E00"
OI_GREEN      <- "#009E73"
OI_SKY        <- "#56B4E9"   # crossfit (matches main AUC figure)
# Sequential palette for one-sided p-values (white → red).
SEQ_STOPS <- c("#FFFFFF", "#FFE5D9", "#FFC2A8", "#FB8161", "#D6604D", "#B2182B")

NLP_THRESHOLD <- -log10(0.05)   # ≈ 1.30; FDR significance cutoff
NLP_VMAX      <- 10             # display ceiling: p_fdr = 1e-10

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
read_slice <- function(model, orient) {
  f <- paste0(CHI2_DIR, model, "_", orient, ".csv")
  m <- as.matrix(read.csv(f, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

read_template <- function(orient) {
  f <- paste0(TPL_DIR, "template_", orient, ".csv")
  m <- as.matrix(read.csv(f, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

# All four methods. Subsets are taken below for the main vs appendix figures.
all_models    <- c("base_bal", "base_conf", "refit", "crossfit")
all_model_lbl <- c(base_bal  = "No Control",
                   base_conf = "No Control",
                   refit     = "Age Control",
                   crossfit  = "Age Control")
all_model_sub <- c(base_bal  = "(Balanced training)",
                   base_conf = "(Confounded training)",
                   refit     = "Sample-split, Confounded",
                   crossfit  = "Cross-fit, Confounded")
all_model_col <- c(base_bal  = OI_VERMILLION,
                   base_conf = OI_VERMILLION,
                   refit     = OI_GREEN,
                   crossfit  = OI_SKY)
orients   <- c("sagittal", "coronal", "axial")

# ---------------------------------------------------------------------------
# Panel factories — gray template underlay + thresholded p-value overlay.
# ---------------------------------------------------------------------------
panel_theme <- theme_void() +
  theme(
    plot.margin      = margin(0, 0, 0, 0),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = NA, colour = NA)
  )

make_template_panel <- function(template_df) {
  tdf <- template_df
  tdf$gray <- tdf$value / max(tdf$value)
  ggplot(tdf, aes(col, -row, fill = gray)) +
    geom_raster() +
    scale_fill_gradient(low = "white", high = "grey10",
                        limits = c(0, 1), guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme
}

OVERLAY_ALPHA <- 0.85
make_overlay_panel <- function(df) {
  df$value_show <- df$value
  df$value_show[df$value < NLP_THRESHOLD] <- NA   # FDR > 0.05 → transparent
  df$value_show <- pmin(df$value_show, NLP_VMAX)  # cap at display ceiling
  ggplot(df, aes(col, -row, fill = value_show)) +
    geom_raster(alpha = OVERLAY_ALPHA) +
    scale_fill_gradientn(colours = SEQ_STOPS,
                         limits = c(NLP_THRESHOLD, NLP_VMAX),
                         na.value = "transparent",
                         guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background  = element_rect(fill = "transparent", colour = NA)
    )
}

# ---------------------------------------------------------------------------
# Vertical colourbar (left side) labelled "−log₁₀ p_FDR".
# ---------------------------------------------------------------------------
draw_colorbar <- function() {
  n <- 256
  cols <- colorRampPalette(SEQ_STOPS)(n)
  raster_mat <- matrix(rev(cols), ncol = 1)
  grid.raster(raster_mat,
              x = unit(0.55, "npc"), y = unit(0.5, "npc"),
              width = unit(0.18, "cm"), height = unit(0.82, "npc"),
              interpolate = TRUE)
  grid.text(sprintf("≥%g", NLP_VMAX),
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") + unit(0.41, "npc") + unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6))
  grid.text(sprintf("%.2f", NLP_THRESHOLD),
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") - unit(0.41, "npc") - unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6))
  grid.text("−log₁₀ p_FDR\n(p ≤ 0.05)",
            x = unit(0.55, "npc") - unit(0.35, "cm"),
            y = unit(0.5, "npc"),
            rot = 90,
            gp = gpar(fontfamily = "serif", fontsize = 7,
                      fontface = "bold", col = "grey20",
                      lineheight = 1.0))
}

# ---------------------------------------------------------------------------
# Load all slices
# ---------------------------------------------------------------------------
slices <- list()
for (m in all_models) for (o in orients)
  slices[[paste(m, o, sep = "_")]] <- read_slice(m, o)
templates <- setNames(lapply(orients, read_template), orients)

# ---------------------------------------------------------------------------
# Build a panel figure for an arbitrary subset of models.
# ---------------------------------------------------------------------------
build_panel_pdf <- function(out_path, model_subset, fig_width) {
  cairo_pdf(out_path, width = fig_width, height = 5.3,
            pointsize = 8, fallback_resolution = 600)
  grid.newpage()
  ncols <- length(model_subset)
  pushViewport(viewport(layout = grid.layout(
    nrow    = 4,
    ncol    = ncols + 1,
    heights = unit(c(1.05, 1, 1, 1), c("cm", "null", "null", "null")),
    widths  = unit(c(1.1, rep(1, ncols)),
                   c("cm", rep("null", ncols)))
  )))

  # Column headers.
  for (j in seq_along(model_subset)) {
    m <- model_subset[j]
    pushViewport(viewport(layout.pos.row = 1, layout.pos.col = j + 1))
    grid.text(all_model_lbl[m],
              y  = unit(0.62, "npc"),
              gp = gpar(fontfamily = "serif", fontsize = 8,
                        fontface = "bold", col = all_model_col[m]))
    grid.text(all_model_sub[m],
              y  = unit(0.28, "npc"),
              gp = gpar(fontfamily = "serif", fontsize = 7, col = "grey30"))
    popViewport()
  }

  # Colourbar.
  pushViewport(viewport(layout.pos.row = 2:4, layout.pos.col = 1))
  draw_colorbar()
  popViewport()

  # Panels: per orientation × per model.
  for (i in seq_along(orients)) {
    for (j in seq_along(model_subset)) {
      mdl <- model_subset[j]
      orn <- orients[i]
      pushViewport(viewport(layout.pos.row = i + 1, layout.pos.col = j + 1))
      print(make_template_panel(templates[[orn]]), newpage = FALSE)
      print(make_overlay_panel(slices[[paste(mdl, orn, sep = "_")]]),
            newpage = FALSE)
      popViewport()
    }
  }

  popViewport()
  dev.off()
  cat("Wrote", out_path, "\n")
}

# Main figure: 3 columns — DNN (balanced), DNN (confounded), Cross-fit
# (confounded, marginalized). Mirrors the structure of the AUC main figure.
build_panel_pdf(OUT_PATH_MAIN,
                model_subset = c("base_bal", "base_conf", "crossfit"),
                fig_width    = 5.6)

# Appendix figure: all four methods.
build_panel_pdf(OUT_PATH_APPENDIX,
                model_subset = c("base_bal", "base_conf", "refit", "crossfit"),
                fig_width    = 7.16)
