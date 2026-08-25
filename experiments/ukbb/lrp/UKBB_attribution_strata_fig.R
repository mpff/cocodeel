# UKBB attribution panel — three model arms, each a 2x2 field over the (sex, y) strata.
#
# Claim: the image effect of a DNN trained under confounding attends to different
# structure than one trained without it, and the refit of the confounded model
# recovers the unconfounded attribution.
#
# Reads the centre-slice CSVs written by attribution_strata.py. Signed cohort-mean
# Integrated Gradients, each subject scaled to unit in-brain norm before averaging.
#
# Run:
#   Rscript experiments/ukbb/lrp/UKBB_attribution_strata_fig.R [<attribution_dir>] [<orientation>]

library(ggplot2)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
MAP_DIR <- if (length(args) >= 1) args[1] else
  "experiments/ukbb/runs/final_v2_refit/attribution_fold0"
ORIENT <- if (length(args) >= 2) args[2] else "axial"
OUT_DIR <- file.path(MAP_DIR, "graphics")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Mask to voxels where the across-subject t test survives BH-FDR. Useful on
# unsmoothed maps; after spatial smoothing the test passes over most of the brain
# and the mask stops discriminating, so pass "all" there.
SIG_ONLY <- !(length(args) >= 3 && args[3] == "all")

# "abs": |cohort mean| on one scale shared by every panel, as in the original
# recipe. "signed": the signed mean on a per-arm diverging scale, which keeps the
# direction of evidence but is not comparable in gain between arms.
MODE <- if (length(args) >= 4) args[4] else "abs"

OI_VERMILLION <- "#D55E00"
OI_GREEN      <- "#009E73"
OI_BLUE       <- "#0072B2"
# diverging, blue (negative) to red (positive), white at zero
DIV_STOPS <- c("#2166AC", "#67A9CF", "#D1E5F0", "#FFFFFF",
               "#FDDBC7", "#EF8A62", "#B2182B")
# sequential, for |relevance|; matches the existing UKBB LRP panels
SEQ_STOPS <- c("#FFFFFF", "#FFE5D9", "#FFC2A8", "#FB8161", "#D6604D", "#B2182B")

arms     <- c("dnn_unconf", "dnn_conf", "refit")
arm_lbl  <- c(dnn_unconf = "No control",
              dnn_conf   = "No control",
              refit      = "Age control")
arm_sub  <- c(dnn_unconf = "(unconfounded training)",
              dnn_conf   = "(confounded training)",
              refit      = "(confounded training)")
arm_col  <- c(dnn_unconf = OI_BLUE,
              dnn_conf   = OI_VERMILLION,
              refit      = OI_GREEN)
strata   <- list(c(1, 1), c(0, 1), c(1, 0), c(0, 0))   # (sex, y), row-major over y

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
read_slice <- function(path) {
  m <- as.matrix(read.csv(path, header = FALSE))
  df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
  df$value <- as.vector(m)
  df
}

tag <- function(arm, sex, y) sprintf("%s_sex%d_y%d", arm, sex, y)

template <- read_slice(file.path(MAP_DIR, paste0("template_", ORIENT, ".csv")))
in_brain <- template$value > 1e-3

maps <- list()
sigs <- list()
for (a in arms) for (s in strata) {
  k <- tag(a, s[1], s[2])
  maps[[k]] <- read_slice(file.path(MAP_DIR, paste0(k, "_mean_", ORIENT, ".csv")))
  sigs[[k]] <- read_slice(file.path(MAP_DIR, paste0(k, "_sig_", ORIENT, ".csv")))
}

# In "abs" mode every panel shares one limit, so gain is comparable across arms.
# In "signed" mode each arm gets its own, since the refit head's norm is an order
# of magnitude larger and a shared scale would render the two DNN blocks flat.
arm_abs <- function(a) unlist(lapply(strata, function(s)
  abs(maps[[tag(a, s[1], s[2])]]$value[in_brain])))
VMAX <- if (MODE == "abs") {
  setNames(rep(as.numeric(quantile(unlist(lapply(arms, arm_abs)), 0.995, na.rm = TRUE)),
               length(arms)), arms)
} else {
  setNames(vapply(arms, function(a) as.numeric(quantile(arm_abs(a), 0.995, na.rm = TRUE)),
                  numeric(1)), arms)
}
for (a in arms) message(sprintf("VMAX[%s] = %.3e", a, VMAX[a]))

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
panel_theme <- theme_void() +
  theme(plot.margin      = margin(0, 0, 0, 0),
        panel.background = element_rect(fill = "white", colour = NA),
        plot.background  = element_rect(fill = NA, colour = NA))

template_panel <- function() {
  tdf <- template
  tdf$gray <- tdf$value / max(tdf$value)
  ggplot(tdf, aes(col, -row, fill = gray)) +
    geom_raster() +
    scale_fill_gradient(low = "white", high = "grey15", limits = c(0, 1), guide = "none") +
    coord_fixed(expand = FALSE) +
    panel_theme
}

overlay_panel <- function(k, vmax) {
  df <- maps[[k]]
  shown <- in_brain
  if (SIG_ONLY) shown <- shown & (sigs[[k]]$value > 0.5)
  value <- if (MODE == "abs") pmin(abs(df$value), vmax) else
    pmax(pmin(df$value, vmax), -vmax)
  df$value_show <- ifelse(shown, value, NA_real_)
  # fade out weak attribution so the underlying anatomy stays visible
  df$alpha <- pmin(abs(df$value_show) / (0.45 * vmax), 1)
  fill_scale <- if (MODE == "abs")
    scale_fill_gradientn(colours = SEQ_STOPS, limits = c(0, vmax),
                         na.value = "transparent", guide = "none")
  else
    scale_fill_gradientn(colours = DIV_STOPS, limits = c(-vmax, vmax),
                         na.value = "transparent", guide = "none")
  ggplot(df, aes(col, -row, fill = value_show, alpha = alpha)) +
    geom_raster() +
    scale_alpha_identity(guide = "none") +
    fill_scale +
    coord_fixed(expand = FALSE) +
    panel_theme +
    theme(panel.background = element_rect(fill = "transparent", colour = NA),
          plot.background  = element_rect(fill = "transparent", colour = NA))
}

draw_colorbar <- function() {
  cols <- colorRampPalette(if (MODE == "abs") SEQ_STOPS else DIV_STOPS)(256)
  grid.raster(matrix(rev(cols), ncol = 1),
              x = unit(0.35, "npc"), y = unit(0.5, "npc"),
              width = unit(0.18, "cm"), height = unit(0.72, "npc"),
              interpolate = TRUE)
  ticks <- if (MODE == "abs") list(list("max", 0.36), list("0", -0.36)) else
    list(list("+", 0.36), list("0", 0.0), list("−", -0.36))
  for (lab in ticks) {
    grid.text(lab[[1]], x = unit(0.35, "npc") + unit(0.25, "cm"),
              y = unit(0.5 + lab[[2]], "npc"),
              gp = gpar(fontfamily = "serif", fontsize = 6, col = "grey20"))
  }
  grid.text(if (MODE == "abs") expression(group("|", bar(r)[v], "|")) else expression(bar(r)[v]),
            x = unit(0.35, "npc") - unit(0.28, "cm"), y = unit(0.5, "npc"), rot = 90,
            gp = gpar(fontfamily = "serif", fontsize = 8, col = "grey20"))
}

# ---------------------------------------------------------------------------
# Layout: y label column | 3 arms x (sex = 1, 0) | colour bar
# ---------------------------------------------------------------------------
build_pdf <- function(out_path, fig_width, fig_height) {
  cairo_pdf(out_path, width = fig_width, height = fig_height,
            pointsize = 8, fallback_resolution = 600)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    nrow    = 4,
    ncol    = 8,
    heights = unit(c(1.00, 0.42, 1, 1), c("cm", "cm", "null", "null")),
    widths  = unit(c(0.75, rep(1, 6), 1.05),
                   c("cm", rep("null", 6), "cm"))
  )))

  for (j in seq_along(arms)) {
    a <- arms[j]
    cols <- (2 * j) : (2 * j + 1)
    pushViewport(viewport(layout.pos.row = 1, layout.pos.col = cols))
    grid.text(arm_lbl[a], y = unit(0.64, "npc"),
              gp = gpar(fontfamily = "serif", fontsize = 8.5,
                        fontface = "bold", col = arm_col[a]))
    grid.text(arm_sub[a], y = unit(0.26, "npc"),
              gp = gpar(fontfamily = "serif", fontsize = 7, col = "grey30"))
    popViewport()
    for (k in 1:2) {
      sex <- c(1, 0)[k]
      pushViewport(viewport(layout.pos.row = 2, layout.pos.col = cols[k]))
      grid.text(bquote(italic(S) == .(sex)),
                gp = gpar(fontfamily = "serif", fontsize = 7, col = "grey20"))
      popViewport()
    }
  }

  for (i in 1:2) {
    yv <- c(1, 0)[i]
    pushViewport(viewport(layout.pos.row = i + 2, layout.pos.col = 1))
    grid.text(bquote(italic(Y) == .(yv)), rot = 90,
              gp = gpar(fontfamily = "serif", fontsize = 7.5, col = "grey20"))
    popViewport()
  }

  for (j in seq_along(arms)) {
    for (i in 1:2) {
      for (k in 1:2) {
        sex <- c(1, 0)[k]
        yv  <- c(1, 0)[i]
        pushViewport(viewport(layout.pos.row = i + 2, layout.pos.col = 2 * j + k - 1))
        print(template_panel(), newpage = FALSE)
        print(overlay_panel(tag(arms[j], sex, yv), VMAX[[arms[j]]]), newpage = FALSE)
        popViewport()
      }
    }
  }

  pushViewport(viewport(layout.pos.row = 3:4, layout.pos.col = 8))
  draw_colorbar()
  popViewport()

  popViewport()
  dev.off()
  cat("Wrote", out_path, "\n")
}

build_pdf(file.path(OUT_DIR, paste0("Fig_UKBB_attribution_strata_", ORIENT, ".pdf")),
          fig_width = 7.16, fig_height = 3.5)
