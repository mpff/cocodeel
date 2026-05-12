# UKBB LRP-EpsilonPlus attribution panel — "classic" old-notebook style.
#
# Reads the |.|-then-mean fold-aggregated LRP maps produced by
# `compute_lrp_maps.py` and plots them with a single SHARED global max for
# normalisation across the full figure (matches the recipe in the archived
# UKKBB_HighalcAgeSex_Synthetic.ipynb).
#
# Display:
#   - cmap = coolwarm (RdBu reversed) with limits c(0, 1).
#   - All panels normalised by one global max (= max over in-brain voxels of
#     the panels that appear in the figure). Reader compares "weaker vs
#     stronger" attribution across methods directly.
#   - No transparency, no template underlay (matches old notebook). Voxels
#     outside the brain naturally have ≈0 relevance and render as the cool
#     (blue) end of the palette.
#
# Columns (mirrors UKBB_lrp_panel.R):
#   No Control (Balanced training)    — base_bal
#   No Control (Confounded training)  — base_conf
#   Age Control (Cross-fit, Confounded) — crossfit (main + appendix)
#   Age Control (Sample-split, Confounded) — refit (appendix only)
# Rows: sagittal, coronal, axial.
#
# Run:
#   Rscript experiments/ukbb/lrp/UKBB_lrp_classic_panel.R

library(ggplot2)
library(grid)

RUN_DIR <- "experiments/ukbb/runs/2026-04-26_13-16-37_final_v2/"
LRP_DIR <- paste0(RUN_DIR, "lrp/")
OUT_DIR <- paste0(RUN_DIR, "graphics/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_PATH_MAIN     <- paste0(OUT_DIR, "Fig_UKBB_lrp_classic_main.pdf")
OUT_PATH_APPENDIX <- paste0(OUT_DIR, "Fig_UKBB_lrp_classic_appendix.pdf")

OI_VERMILLION <- "#D55E00"
OI_GREEN      <- "#009E73"
OI_SKY        <- "#56B4E9"

# coolwarm: 9 stops from RColorBrewer RdBu reversed.
COOLWARM_STOPS <- c(
  "#053061", "#2166AC", "#4393C3", "#92C5DE",
  "#F7F7F7",
  "#F4A582", "#D6604D", "#B2182B", "#67001F"
)

read_lrp_slice <- function(model, orient) {
  f <- paste0(LRP_DIR, model, "_lrp_", orient, ".csv")
  m <- as.matrix(read.csv(f, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

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

slices <- list()
for (m in all_models) for (o in orients)
  slices[[paste(m, o, sep = "_")]] <- read_lrp_slice(m, o)

# Build a panel figure with a SHARED global max across the displayed panels.
panel_theme <- theme_void() +
  theme(
    plot.margin      = margin(0, 0, 0, 0),
    panel.background = element_rect(fill = "white", colour = NA),
    plot.background  = element_rect(fill = NA, colour = NA)
  )

make_overlay_panel <- function(slice_df, vmax) {
  df <- slice_df
  df$value_show <- pmin(df$value, vmax) / vmax   # → [0, 1]
  ggplot(df, aes(col, -row, fill = value_show)) +
    geom_raster() +
    scale_fill_gradientn(colours = COOLWARM_STOPS,
                         limits = c(0, 1),
                         na.value = "transparent",
                         guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme
}

draw_colorbar <- function() {
  n <- 256
  cols <- colorRampPalette(COOLWARM_STOPS)(n)
  raster_mat <- matrix(rev(cols), ncol = 1)
  grid.raster(raster_mat,
              x = unit(0.55, "npc"), y = unit(0.5, "npc"),
              width = unit(0.18, "cm"), height = unit(0.82, "npc"),
              interpolate = TRUE)
  grid.text("max",
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") + unit(0.41, "npc") + unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6, col = "grey20"))
  grid.text("0",
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") - unit(0.41, "npc") - unit(0.15, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6, col = "grey20"))
  grid.text("Mean |LRP relevance|\n(per-method scale)",
            x = unit(0.55, "npc") - unit(0.35, "cm"),
            y = unit(0.5, "npc"),
            rot = 90,
            gp = gpar(fontfamily = "serif", fontsize = 7,
                      fontface = "bold", col = "grey20",
                      lineheight = 1.0))
}

build_panel_pdf <- function(out_path, model_subset, fig_width) {
  # Per-method scale: each method normalised by its own max (pooled across
  # the three orientations of that method). Avoids the post-hoc max
  # dominating the base panels when the relevance scales differ ~5×.
  vmax_per_model <- setNames(
    vapply(model_subset, function(m) {
      pooled <- unlist(lapply(orients, function(o) {
        slices[[paste(m, o, sep = "_")]]$value
      }))
      max(pooled, na.rm = TRUE)
    }, numeric(1)),
    model_subset
  )
  for (m in model_subset)
    message(sprintf("[%s] vmax[%s] = %.3e",
                    basename(out_path), m, vmax_per_model[m]))

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

  pushViewport(viewport(layout.pos.row = 2:4, layout.pos.col = 1))
  draw_colorbar()
  popViewport()

  for (i in seq_along(orients)) {
    for (j in seq_along(model_subset)) {
      mdl <- model_subset[j]
      orn <- orients[i]
      pushViewport(viewport(layout.pos.row = i + 1, layout.pos.col = j + 1))
      print(make_overlay_panel(slices[[paste(mdl, orn, sep = "_")]],
                               vmax_per_model[[mdl]]),
            newpage = FALSE)
      popViewport()
    }
  }

  popViewport()
  dev.off()
  cat("Wrote", out_path, "\n")
}

build_panel_pdf(OUT_PATH_MAIN,
                model_subset = c("base_bal", "base_conf", "crossfit"),
                fig_width    = 5.6)

build_panel_pdf(OUT_PATH_APPENDIX,
                model_subset = c("base_bal", "base_conf", "refit", "crossfit"),
                fig_width    = 7.16)
