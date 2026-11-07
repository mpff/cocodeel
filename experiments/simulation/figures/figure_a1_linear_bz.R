library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(png)
library(grid)
library(patchwork)

# Load increasing fz dataset

effect_names <- c(
  'y' = 'y',
  'fx' = 'Image Effect',
  'fr' = 'Residual Image Effect',
  'fz' = 'Covariate Effect'
)

df_bz <- read_csv("experiments/simulation/output/increasing_bz.csv") %>%
  mutate(model = factor(
    model,
    levels= c( "refit", "refit_orth", "base", "posthoc_web"),
    labels = c(
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "DNN (Baseline)\n+ Orthogonalisation")
  )) %>%
  filter(n < 50000, !is.na(model)) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))


# Load example image
img <- readPNG("experiments/simulation/output/example_image.png")
g <- rasterGrob(img, interpolate=TRUE)


### PLOT ###

# a.    | b. DE | 
# Graph | ----- |
#       | c. RE |

plot_params <- list(
  list("metric" = "mspe", "yname" = TeX("$MSPE(\\hat{f}_X)$"), "outname" = "pfeuffer1.pdf"),
  list("metric" = "bias2", "yname" = TeX("$Bias^2(\\hat{f}_X)$"), "outname" = "pfeufferS1a.pdf"),
  list("metric" = "var", "yname" = TeX("$Var(\\hat{f}_X)$"), "outname" = "pfeufferS1b.pdf")
)


# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.5, 0.9),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit="inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    text = element_text(size = 8, family = "serif"),
    #axis.title.x = element_text(vjust = 1),
    #axis.title.y = element_text(hjust = 0),
    plot.margin = margin(2, 2, 2, 2)   # reduce outer whitespace
    #panel.spacing = unit(1, "pt")       # spacing between facet panels
  )

# Reusable plotting function
  make_plot <- function(data, ylab, color_option, show_legend = TRUE, strip_labels = TRUE,
                        ylim = c(.4 * 1e-4, 1.25), xlim = c(175, 100 * 2^7.05),
                        vertical = TRUE, legend.pos = c(0.5, 0.9)) {
    p <- ggplot(
      data,
      aes(x = n * 0.5, y = value, group = bz)
    ) +
      #geom_hline(yintercept = 0, linewidth = 0.5, color = "grey80") +
      geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
      geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
      scale_x_log10(
        name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
        breaks = 100 * 2^(1:7)
      ) +
      scale_y_log10(name = ylab) +
      scale_color_viridis_c(
        begin = 0,
        end = 1,
        option = color_option,
        limits = c(-0.025, 4.25),
        breaks = c(0,1,2,3,4),
        name = TeX("$\\beta_Z$")
      ) +
      coord_cartesian(
        xlim = xlim,
        ylim = ylim
      ) +
      shared_theme +
      theme(legend.position = legend.pos)
    
    if (vertical) {
      p <- p + facet_grid(model ~ .)
    } else {
      p <- p + facet_grid(. ~ model)
    }
    
    if (!show_legend) {
      p <- p + theme(legend.position = "none")
    }
    if (strip_labels) {
      p <- p + theme(strip.text.y = element_blank())
    }
    
    p
  }

# Build plots
b3 <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$MSPE(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE,
  ylim = c(.4 * 1e-3, 1.25)
)

c3 <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = TRUE,
  strip_labels = FALSE,
  ylim = c(.4 * 1e-3, 1.25)
)


d <- b3 + c3

ggsave("experiments/simulation/output/graphics/Fig1b_bz_mspe.pdf", d, width = 4.16, height = 2.4, units = "in", dpi=600, device = cairo_pdf)


fxBias <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "bias2",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Bias^2(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE,
  vertical = FALSE,
  legend.pos = c(0.3, 0.8)
)

fxVar <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "var",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Var(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = FALSE,
  strip_labels = FALSE,
  vertical = FALSE
)

S1a <- fxBias / fxVar

ggsave("experiments/simulation/output/graphics/FigS1a_bz_bias2.pdf", S1a, width = 3.5, height = 3.5, units = "in", dpi=600, device = cairo_pdf)


frBias <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "bias2",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Bias^2(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = TRUE,
  strip_labels = FALSE,
  vertical = FALSE,
  legend.pos = c(0.3, 0.8)
)

frVar <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "var",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Var(\\hat{f}^{re}_X)$$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = FALSE,
  strip_labels = FALSE,
  vertical = FALSE
)

S1b <- frBias / frVar

ggsave("experiments/simulation/output/graphics/FigS1b_bz_var.pdf", S1b, width = 3.5, height = 3.5, units = "in", dpi=600, device = cairo_pdf)



### PLOT ###

# a.    | b. DE | 
# Graph | ----- |
#       | c. RE |


# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.5, 0.9),
    legend.direction = "horizontal",
    legend.margin = margin(0, unit="inch"),
    legend.background = element_rect(color = NA, fill = NA),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size=6),
    axis.text.y = element_text(hjust = 1.25, size=6),
    text = element_text(size = 8, family = "serif"),
    #axis.title.x = element_text(vjust = 1),
    #axis.title.y = element_text(hjust = 0),
    plot.margin = margin(2, 2, 2, 2)   # reduce outer whitespace
    #panel.spacing = unit(1, "pt")       # spacing between facet panels
  )

# Reusable plotting function for the fz panels.
#
# Methods that do not estimate fz (base, posthoc_web) emit a constant
# zero, producing a flat horizontal line at the variance of fz_truth
# (~bz²/12). Plotting them with the bz-viridis gradient is misleading —
# the lines are not informative about model behaviour. The
# `estimators` argument names which models DO estimate fz; only those
# are drawn with the colour gradient. All other rows are still drawn
# but in grey, signalling "this method does not estimate fz".
make_plot <- function(data, ylab, color_option, estimators,
                      show_legend = TRUE, strip_labels = TRUE,
                      vertical = TRUE, legend.pos = c(0.5, 0.9),
                      ylim = c(1e-5, 1.25)) {
  p <- ggplot(
    data,
    aes(x = n * 0.5, y = value, group = bz)
  ) +
    geom_line(color = "grey75", alpha = 0.6, linewidth = 0.6) +
    geom_point(color = "grey75", alpha = 0.6, size = 0.6) +
    geom_line(data = ~ filter(.x, model %in% estimators),
              aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(data = ~ filter(.x, model %in% estimators),
               aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(1:7)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_viridis_c(
      begin = 0,
      end = 1,
      option = color_option,
      limits = c(-0.025, 4.25),
      breaks = c(0,1,2,3,4),
      name = TeX("$\\beta_Z$")
    ) +
    coord_cartesian(
      xlim = c(175, 100 * 2^7.5),
      ylim = ylim
    ) +
    shared_theme +
    theme(legend.position = legend.pos)

  if (vertical) {
    p <- p + facet_grid(model ~ .)
  } else {
    p <- p + facet_grid(. ~ model)
  }

  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }

  p
}

# Build plots. `estimators` names the methods that ACTUALLY estimate
# fz (vs. emit a constant 0); only these get the bz-viridis colouring,
# the others are drawn in grey.
ESTIMATORS_DIRECT <- "DNN with Controls"
ESTIMATORS_ORTH   <- "DNN with Controls\n+ Orthogonalisation"

b1 <- make_plot(
  df_bz %>% filter(
    effect == "fz",
    metric == "bias2",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Bias^2(\\hat{f}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  estimators = ESTIMATORS_DIRECT,
  show_legend = TRUE,
  strip_labels = FALSE
)

c1 <- make_plot(
  df_bz %>% filter(
    effect == "fz",
    metric == "bias2",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Bias^2(\\hat{f}^{tot}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  estimators = ESTIMATORS_ORTH,
  show_legend = TRUE,
  strip_labels = FALSE
)
b2 <- make_plot(
  df_bz %>% filter(
    effect == "fz",
    metric == "var",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Var(\\hat{f}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  estimators = ESTIMATORS_DIRECT,
  show_legend = FALSE,
  strip_labels = FALSE
)

c2 <- make_plot(
  df_bz %>% filter(
    effect == "fz",
    metric == "var",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Var(\\hat{f}^{tot}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  estimators = ESTIMATORS_ORTH,
  show_legend = FALSE,
  strip_labels = FALSE
)

b <- b1 + c1 
#+ plot_annotation(
#  title = "b. Estimation of the direct X-effect", 
#  theme = theme(plot.title = element_text(size = 9, family = "serif"))
#  )
c <- b2 + c2
#+ plot_annotation(
#  title = "c. Estimation of the residual X-effect",
#  theme = theme(plot.title = element_text(size = 9, family = "serif"))
#)

b/c

ggsave("experiments/simulation/output/graphics/FigS1b_fz_bz_bias.pdf", b, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)
ggsave("experiments/simulation/output/graphics/FigS1c_fz_bz_var.pdf", c, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)


### Bias^2 and Var of f_X / f_X^re, MSPE(yhat) and MSPE(fzhat) in one appendix figure

UNORTH <- c("DNN with Controls", "DNN (Baseline)")
ORTH   <- c("DNN with Controls\n+ Orthogonalisation", "DNN (Baseline)\n+ Orthogonalisation")

theme_cell <- theme_bw() + theme(
  text = element_text(size = 8, family = "serif"),
  axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1.2, size = 6),
  axis.text.y = element_text(size = 6),
  strip.text = element_text(size = 6.5),
  plot.title = element_text(size = 8, hjust = 0.5),
  legend.title = element_text(size = 8), legend.text = element_text(size = 6),
  legend.key.height = unit(0.07, "in"), legend.key.width = unit(0.25, "in"),
  legend.direction = "horizontal", legend.title.position = "left",
  legend.ticks = element_line(colour = "black"),
  plot.margin = margin(3, 3, 3, 3))

# one panel per (model, metric); every panel is its own plot so that patchwork spaces them evenly
LOG_LABELS <- function(x) formatC(x, format = "e", digits = 0)
LIN_LABELS <- function(x) paste0("\u2007\u2007", formatC(x, format = "f", digits = 1))   # figure spaces pad to the width of "1e-04"
COLUMNS <- list(
  list(metric = "bias2", effect = "fx", title = "$Bias^2(\\hat{f}_X)$, $Bias^2(\\hat{f}_X^{re})$", ylim = c(4e-5, 1.25)),
  list(metric = "var",   effect = "fx", title = "$Var(\\hat{f}_X)$, $Var(\\hat{f}_X^{re})$",       ylim = c(4e-5, 1.25)),
  list(metric = "mspe",  effect = "y",  title = "$MSPE(\\hat{y})$",                                ylim = c(0.95, 5.7)),
  list(metric = "mspe",  effect = "fz", title = "$MSPE(\\hat{f}_Z)$",                              ylim = c(1e-5, 1.25))
)
# rows: (model, effect used for the f_X columns, palette); orthogonalised models target f_X^re
ROWS <- list(
  list(model = UNORTH[1], fx = "fx", palette = "viridis"),
  list(model = UNORTH[2], fx = "fx", palette = "viridis"),
  list(model = ORTH[1],   fx = "fr", palette = "inferno"),
  list(model = ORTH[2],   fx = "fr", palette = "inferno")
)
FZ_ESTIMATORS <- c(UNORTH[1], ORTH[1])   # the others emit f_Z = 0 and are drawn in grey

panel <- function(row, col, top, bottom, legend) {
  d <- df_bz %>% filter(model == row$model, metric == col$metric,
                        effect == if (col$effect == "fx") row$fx else col$effect)
  grey <- col$effect == "fz" && !(row$model %in% FZ_ESTIMATORS)
  p <- ggplot(d, aes(n * 0.5, value, group = bz))
  if (col$effect == "y")
    p <- p + geom_hline(yintercept = 1, linetype = "dashed", colour = "grey40", linewidth = 0.4)
  p <- p + if (grey) list(geom_line(colour = "grey75", linewidth = 0.6), geom_point(colour = "grey75", size = 0.6))
           else list(geom_line(aes(colour = bz), alpha = 0.8, linewidth = 0.7),
                     geom_point(aes(colour = bz), alpha = 0.8, size = 0.6))
  p <- p +
    scale_x_log10(name = if (bottom) TeX("$N_{train}$") else NULL, breaks = 100 * 4^(1:4) / 2) +
    scale_colour_viridis_c(option = row$palette, limits = c(-0.025, 4.25), breaks = 0:4,
                           name = TeX("$\\beta_Z$"), guide = if (legend) "colourbar" else "none") +
    (if (col$effect == "y") scale_y_continuous(breaks = 1:5, labels = LIN_LABELS)
     else scale_y_log10(breaks = 10^c(-4, -2, 0), labels = LOG_LABELS)) +
    coord_cartesian(xlim = c(175, 100 * 2^7.05), ylim = col$ylim) +
    labs(y = NULL, title = if (top) TeX(col$title) else NULL) + theme_cell
  if (!bottom) p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  p
}
# model name in a grey box of the same size in every row, as a separate column on the right
label_box <- function(row) {
  ggplot() +
    annotate("text", x = 0, y = 0, label = row$model, angle = -90, size = 6.5 / .pt, family = "serif") +
    theme_void() +
    theme(panel.background = element_rect(fill = "grey85", colour = "grey20", linewidth = 0.3),
          plot.margin = margin(3, 3, 3, 0))
}
panels <- list()
for (i in seq_along(ROWS)) {
  for (j in seq_along(COLUMNS))
    panels[[length(panels) + 1]] <- panel(ROWS[[i]], COLUMNS[[j]], top = i == 1, bottom = i == 4,
                                          legend = j == 1 && i %in% c(1, 3))
  panels[[length(panels) + 1]] <- label_box(ROWS[[i]])
}
fig <- wrap_plots(panels, ncol = 5, byrow = TRUE) +
  plot_layout(guides = "collect", widths = c(1, 1, 1, 1, 0.22)) & theme(legend.position = "bottom")
ggsave("experiments/simulation/output/graphics/Fig_bz_appendix.pdf", fig,
       width = 5.5, height = 5.2, units = "in", device = cairo_pdf)
