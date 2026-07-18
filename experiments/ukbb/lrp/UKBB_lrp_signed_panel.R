# UKBB LRP signed-attribution panel — divergent (coolwarm) overlay style.
#
# Plots the cohort-mean signed IG attribution per voxel:
#     r̄_v = (1/n) Σ_i r̄_{i,v}        (n = 100 subjects, sex==1 ∧ y==1)
# with each subject's r̄_{i,v} averaged over 5 folds. Output of
# compute_chi2_maps.py (the `<model>_mean_<orient>.csv` slice dumps).
#
# Display:
#   - Divergent RdBu_r palette (blue ← negative · zero · positive → red).
#   - Symmetric limits c(-vmax, +vmax), vmax = 99.5th percentile of |r̄|
#     in-brain across all displayed methods (so panels share a scale).
#   - No transparency — overlay is opaque inside the brain mask, NA outside.
#   - Gray template brain visible only outside the brain mask, providing
#     contour/skull context.
#
# Columns (mirrors UKBB_lrp_panel.R):
#   No Control (Balanced training)    — base_bal
#   No Control (Confounded training)  — base_conf
#   Age Control (Confounded training) — crossfit (main figure)
#   Age Control (Confounded, sample-split refit) — refit (appendix only)
# Rows: sagittal, coronal, axial.
#
# Run:
#   Rscript experiments/ukbb/lrp/UKBB_lrp_signed_panel.R

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
OUT_PATH_MAIN     <- paste0(OUT_DIR, "Fig_UKBB_lrp_signed_main.pdf")
OUT_PATH_APPENDIX <- paste0(OUT_DIR, "Fig_UKBB_lrp_signed_appendix.pdf")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
OI_VERMILLION <- "#D55E00"
OI_GREEN      <- "#009E73"
OI_SKY        <- "#56B4E9"
# Sequential warm palette for |r̄| (low → high). White → orange → red.
SEQ_STOPS <- c("#FFFFFF", "#FFE5D9", "#FFC2A8", "#FB8161", "#D6604D", "#B2182B")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
read_signed_slice <- function(model, orient) {
  f <- paste0(CHI2_DIR, model, "_mean_", orient, ".csv")
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

# All four methods.
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
orients       <- c("sagittal", "coronal", "axial")

# ---------------------------------------------------------------------------
# Load all slices.
# ---------------------------------------------------------------------------
slices <- list()
for (m in all_models) for (o in orients)
  slices[[paste(m, o, sep = "_")]] <- read_signed_slice(m, o)
templates <- setNames(lapply(orients, read_template), orients)

# Per-method symmetric colour limit (99.5th pctl of |r̄| in-brain, pooled
# across the three orientations for that method). Each method renders on its
# own scale — standard LRP-paper convention. The base DNN's logit-scale
# attribution and the refit fx-scale attribution differ by ~5×, so a
# shared VMAX would wash out the base panels.
brain_abs <- function(slice_df, tpl_df) {
  in_brain <- tpl_df$value > 1e-3
  abs(slice_df$value[in_brain])
}
VMAX_PER_MODEL <- setNames(
  vapply(all_models, function(m) {
    pooled <- unlist(lapply(orients, function(o) {
      brain_abs(slices[[paste(m, o, sep = "_")]], templates[[o]])
    }))
    as.numeric(quantile(pooled, 0.995, na.rm = TRUE))
  }, numeric(1)),
  all_models
)
for (m in all_models) message(sprintf("VMAX[%s] = ±%.3e", m, VMAX_PER_MODEL[m]))

# ---------------------------------------------------------------------------
# Panel factories.
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
    scale_fill_gradient(low = "white", high = "grey15",
                        limits = c(0, 1), guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme
}

# Overlay: opaque |r̄| heat-map inside brain, transparent outside.
make_overlay_panel <- function(slice_df, template_df, vmax) {
  df <- slice_df
  df$brain <- template_df$value > 1e-3
  abs_value <- pmin(abs(df$value), vmax)
  df$value_show <- ifelse(df$brain, abs_value, NA_real_)
  ggplot(df, aes(col, -row, fill = value_show)) +
    geom_raster() +
    scale_fill_gradientn(colours = SEQ_STOPS,
                         limits = c(0, vmax),
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
# Vertical colour-bar (signed, divergent).
# ---------------------------------------------------------------------------
draw_colorbar <- function() {
  n <- 256
  cols <- colorRampPalette(SEQ_STOPS)(n)
  raster_mat <- matrix(rev(cols), ncol = 1)
  grid.raster(raster_mat,
              x = unit(0.55, "npc"), y = unit(0.5, "npc"),
              width = unit(0.18, "cm"), height = unit(0.82, "npc"),
              interpolate = TRUE)
  grid.text("high",
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") + unit(0.41, "npc") + unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6, col = "grey20"))
  grid.text("0",
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") - unit(0.41, "npc") - unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6, col = "grey20"))
  grid.text("|Mean IG attribution|\n(per-method scale)",
            x = unit(0.55, "npc") - unit(0.35, "cm"),
            y = unit(0.5, "npc"),
            rot = 90,
            gp = gpar(fontfamily = "serif", fontsize = 7,
                      fontface = "bold", col = "grey20",
                      lineheight = 1.0))
}

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

  # Colour-bar.
  pushViewport(viewport(layout.pos.row = 2:4, layout.pos.col = 1))
  draw_colorbar()
  popViewport()

  # Per orientation × per model.
  for (i in seq_along(orients)) {
    for (j in seq_along(model_subset)) {
      mdl <- model_subset[j]
      orn <- orients[i]
      pushViewport(viewport(layout.pos.row = i + 1, layout.pos.col = j + 1))
      print(make_template_panel(templates[[orn]]), newpage = FALSE)
      print(make_overlay_panel(slices[[paste(mdl, orn, sep = "_")]],
                               templates[[orn]],
                               vmax = VMAX_PER_MODEL[[mdl]]),
            newpage = FALSE)
      popViewport()
    }
  }

  popViewport()
  dev.off()
  cat("Wrote", out_path, "\n")
}

# Main: 3 cols — base_bal, base_conf, crossfit (cross-fit confounded).
build_panel_pdf(OUT_PATH_MAIN,
                model_subset = c("base_bal", "base_conf", "crossfit"),
                fig_width    = 5.6)

# Appendix: 4 cols — adds the sample-split refit.
build_panel_pdf(OUT_PATH_APPENDIX,
                model_subset = c("base_bal", "base_conf", "refit", "crossfit"),
                fig_width    = 7.16)
