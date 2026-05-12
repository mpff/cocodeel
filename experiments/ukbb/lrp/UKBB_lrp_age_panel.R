# Age-stratified χ² panel — single sagittal slice, 6 panels in a row.
#
# Layout: 1 row × 6 cols, grouped young/old PAIR per model:
#   [Bal Y][Bal O] | [Conf Y][Conf O] | [Refit Y][Refit O]
#
# Reads from:
#   chi2_young/<model>_sagittal.csv
#   chi2_old  /<model>_sagittal.csv
#   lrp_maps/template_sagittal.csv
#
# Plots −log₁₀ p_FDR per voxel, thresholded at p_FDR ≤ 0.05.
#
# Claim: if the confounded model attends to age-correlated anatomy, the
# significance pattern should differ between Young and Old for `base_conf`;
# for the post-hoc `refit` the patterns should look more similar across
# strata, since the linear age term has absorbed the age-related
# contribution. `base_bal` (no confound injected) is the reference for
# "what natural age-pattern variation looks like".
#
# Run:
#   Rscript experiments/ukbb/lrp/UKBB_lrp_age_panel.R

library(ggplot2)
library(grid)

RUN_DIR  <- "experiments/ukbb/runs/2026-04-14_17-26-52_final/"
TPL_DIR  <- paste0(RUN_DIR, "lrp_maps/")
OUT_DIR  <- paste0(RUN_DIR, "graphics/")
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_PATH <- paste0(OUT_DIR, "Fig_UKBB_lrp_age_strata.pdf")

OI_VERMILLION <- "#D55E00"
OI_GREEN      <- "#009E73"
SEQ_STOPS     <- c("#FFFFFF", "#FFE5D9", "#FFC2A8", "#FB8161", "#D6604D", "#B2182B")
NLP_THRESHOLD <- -log10(0.05)
NLP_VMAX      <- 10

# Display order: per model, young then old (so within-model age contrast is
# adjacent — easiest read for the claim).
panels <- list(
  list(model = "base_bal",  stratum = "young", group = "No Control (Bal)",   group_col = OI_VERMILLION),
  list(model = "base_bal",  stratum = "old",   group = "No Control (Bal)",   group_col = OI_VERMILLION),
  list(model = "base_conf", stratum = "young", group = "No Control (Conf)",  group_col = OI_VERMILLION),
  list(model = "base_conf", stratum = "old",   group = "No Control (Conf)",  group_col = OI_VERMILLION),
  list(model = "refit",     stratum = "young", group = "Age Control (Conf)", group_col = OI_GREEN),
  list(model = "refit",     stratum = "old",   group = "Age Control (Conf)", group_col = OI_GREEN)
)
stratum_lbl <- c(young = "Young", old = "Old")

read_slice <- function(stratum, model) {
  f <- paste0(RUN_DIR, "chi2_", stratum, "/", model, "_sagittal.csv")
  m <- as.matrix(read.csv(f, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

read_template <- function() {
  f <- paste0(TPL_DIR, "template_sagittal.csv")
  m <- as.matrix(read.csv(f, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

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
    coord_fixed(expand = FALSE) + panel_theme
}

OVERLAY_ALPHA <- 0.85
make_overlay_panel <- function(df) {
  df$value_show <- df$value
  df$value_show[df$value < NLP_THRESHOLD] <- NA
  df$value_show <- pmin(df$value_show, NLP_VMAX)
  ggplot(df, aes(col, -row, fill = value_show)) +
    geom_raster(alpha = OVERLAY_ALPHA) +
    scale_fill_gradientn(colours = SEQ_STOPS,
                         limits = c(NLP_THRESHOLD, NLP_VMAX),
                         na.value = "transparent", guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background  = element_rect(fill = "transparent", colour = NA)
    )
}

draw_colorbar <- function() {
  n <- 256
  cols <- colorRampPalette(SEQ_STOPS)(n)
  raster_mat <- matrix(rev(cols), ncol = 1)
  grid.raster(raster_mat,
              x = unit(0.55, "npc"), y = unit(0.5, "npc"),
              width = unit(0.18, "cm"), height = unit(0.78, "npc"),
              interpolate = TRUE)
  grid.text(sprintf("≥%g", NLP_VMAX),
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") + unit(0.39, "npc") + unit(0.13, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6))
  grid.text(sprintf("%.2f", NLP_THRESHOLD),
            x = unit(0.55, "npc"),
            y = unit(0.5, "npc") - unit(0.39, "npc") - unit(0.13, "cm"),
            gp = gpar(fontfamily = "serif", fontsize = 6))
  grid.text("−log₁₀ p_FDR\n(p ≤ 0.05)",
            x = unit(0.55, "npc") - unit(0.32, "cm"),
            y = unit(0.5, "npc"),
            rot = 90,
            gp = gpar(fontfamily = "serif", fontsize = 7,
                      fontface = "bold", col = "grey20",
                      lineheight = 1.0))
}

# --- Load data --------------------------------------------------------------
slices <- lapply(panels, function(p) read_slice(p$stratum, p$model))
template_df <- read_template()

# --- Layout ----------------------------------------------------------------
# 3 rows (group header / stratum subhead / panel) × 7 cols (colourbar + 6 slices)
cairo_pdf(OUT_PATH, width = 7.16, height = 1.95,
          pointsize = 8, fallback_resolution = 600)

grid.newpage()
n_cols <- 1 + length(panels)
pushViewport(viewport(layout = grid.layout(
  nrow = 3, ncol = n_cols,
  heights = unit(c(0.55, 0.40, 1), c("cm", "cm", "null")),
  widths  = unit(c(1.1, rep(1, length(panels))),
                 c("cm", rep("null", length(panels))))
)))

# --- Group headers (span 2 cols per group) ---------------------------------
group_starts <- c(2, 4, 6)            # first panel-col per group (after colourbar)
group_lbls   <- c("No Control", "No Control", "Age Control")
group_subs   <- c("(Balanced)",  "(Confounded)", "(Confounded)")
group_cols   <- c(OI_VERMILLION, OI_VERMILLION, OI_GREEN)

for (k in seq_along(group_starts)) {
  c0 <- group_starts[k]
  pushViewport(viewport(layout.pos.row = 1, layout.pos.col = c(c0, c0 + 1)))
  grid.text(group_lbls[k],
            y  = unit(0.68, "npc"),
            gp = gpar(fontfamily = "serif", fontsize = 8,
                      fontface = "bold", col = group_cols[k]))
  grid.text(group_subs[k],
            y  = unit(0.22, "npc"),
            gp = gpar(fontfamily = "serif", fontsize = 7, col = "grey30"))
  popViewport()
}

# --- Stratum sub-labels (one per panel col) --------------------------------
for (j in seq_along(panels)) {
  pushViewport(viewport(layout.pos.row = 2, layout.pos.col = j + 1))
  grid.text(stratum_lbl[panels[[j]]$stratum],
            gp = gpar(fontfamily = "serif", fontsize = 7,
                      fontface = "italic", col = "grey25"))
  popViewport()
}

# --- Colourbar (full panel-row height) -------------------------------------
pushViewport(viewport(layout.pos.row = 3, layout.pos.col = 1))
draw_colorbar()
popViewport()

# --- Panels ----------------------------------------------------------------
for (j in seq_along(panels)) {
  pushViewport(viewport(layout.pos.row = 3, layout.pos.col = j + 1))
  print(make_template_panel(template_df), newpage = FALSE)
  print(make_overlay_panel(slices[[j]]), newpage = FALSE)
  popViewport()
}

popViewport()
dev.off()
cat("Wrote", OUT_PATH, "\n")
