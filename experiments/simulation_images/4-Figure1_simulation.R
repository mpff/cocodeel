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

df_bz <- read_csv("results/simulation_images/increasing_bz.csv") %>%
  mutate(model = factor(
    model,
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN with Controls",
      "DNN with Controls\n+ Orthogonalisation",
      "DNN (Baseline)",
      "[17] DNN (Baseline)\n+ Orthogonalisation")
  )) %>%
  filter(n < 50000, !is.na(model)) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))


# Load example image
img <- readPNG("results/simulation_images/example_image.png")
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
b1 <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "bias2",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Bias^2(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE
)

c1 <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "bias2",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Bias^2(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = TRUE,
  strip_labels = FALSE
  )



b2 <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "var",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$Var(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = FALSE,
  strip_labels = FALSE
)

c2 <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "var",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Var(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = FALSE,
  strip_labels = FALSE
)


b3 <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN (Baseline)", "DNN with Controls")
  ),
  ylab = TeX("$MSPE^2(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE,
  ylim = c(.4 * 1e-3, 1.25)
)

c3 <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$MSPE(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = TRUE,
  strip_labels = FALSE,
  ylim = c(.4 * 1e-3, 1.25)
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
d <- b3 + c3

d/b/c

ggsave("graphics/Fig1b_bz_mspe.pdf", d, width = 4.16, height = 2.4, units = "in", dpi=600, device = cairo_pdf)
ggsave("graphics/Fig1b_bz_bias2.pdf", b, width = 4.16, height = 2.4, units = "in", dpi=600, device = cairo_pdf)
ggsave("graphics/Fig1b_bz_var.pdf", c, width = 4.16, height = 2.4, units = "in", dpi=600, device = cairo_pdf)


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

ggsave("graphics/FigS1a_bz_bias2.pdf", S1a, width = 3.5, height = 3.5, units = "in", dpi=600, device = cairo_pdf)


frBias <- make_plot(
  df_bz %>% filter(
    effect == "fr",
    metric == "bias2",
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
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
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
  ),
  ylab = TeX("$Var(\\hat{f}^{re}_X)$$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = FALSE,
  strip_labels = FALSE,
  vertical = FALSE
)

S1b <- frBias / frVar

ggsave("graphics/FigS1b_bz_var.pdf", S1b, width = 3.5, height = 3.5, units = "in", dpi=600, device = cairo_pdf)



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
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
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
    model %in% c("DNN with Controls\n+ Orthogonalisation", "[17] DNN (Baseline)\n+ Orthogonalisation")
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

ggsave("graphics/FigS1b_fz_bz_bias.pdf", b, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)
ggsave("graphics/FigS1c_fz_bz_var.pdf", c, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)


### MSPE(yhat) and MSPE(fzhat) — two FigS1a-style PDFs (one unorth, one orth)

UNORTH <- c("DNN (Baseline)", "DNN with Controls")
ORTH   <- c("DNN with Controls\n+ Orthogonalisation",
            "[17] DNN (Baseline)\n+ Orthogonalisation")

# Inline plotter for MSPE(yhat): values sit in [1.003, 5.47] for the four
# plotted methods, so a linear scale starting near the noise floor at 1.0
# resolves the across-bz spread that a log scale crushes against y=1.
# Mirrors `make_plot`'s API: vertical = FALSE -> facet_grid(. ~ model).
make_y_plot <- function(data, color_option, show_legend = TRUE,
                        legend.pos = c(0.3, 0.8),
                        ylim = c(0.95, 5.7), vertical = TRUE) {
  p <- ggplot(data, aes(x = n * 0.5, y = value, group = bz)) +
    geom_hline(yintercept = 1, linetype = "dashed",
               color = "grey40", linewidth = 0.4) +
    geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
                  breaks = 100 * 2^(1:7)) +
    scale_y_continuous(name = TeX("$MSPE(\\hat{y})$"),
                       breaks = c(1, 2, 3, 4, 5)) +
    scale_color_viridis_c(begin = 0, end = 1, option = color_option,
                          limits = c(-0.025, 4.25),
                          breaks = c(0, 1, 2, 3, 4),
                          name = TeX("$\\beta_Z$")) +
    coord_cartesian(xlim = c(175, 100 * 2^7.05), ylim = ylim) +
    shared_theme +
    theme(legend.position = legend.pos)
  if (vertical) p <- p + facet_grid(model ~ .)
  else          p <- p + facet_grid(. ~ model)
  if (!show_legend) p <- p + theme(legend.position = "none")
  p
}

# MSPE(yhat): UNORTH (left, viridis) | ORTH (right, inferno).
# ylim top = 7 leaves headroom above the highest curve (~5.4) so the
# default legend at panel-y 0.9 sits in empty space.
y_b <- make_y_plot(
  df_bz %>% filter(effect == "y", metric == "mspe", model %in% UNORTH),
  color_option = "viridis",
  show_legend = TRUE,
  legend.pos = c(0.5, 0.9),
  ylim = c(0.95, 7),
  vertical = TRUE
)
y_c <- make_y_plot(
  df_bz %>% filter(effect == "y", metric == "mspe", model %in% ORTH),
  color_option = "inferno",
  show_legend = TRUE,
  legend.pos = c(0.5, 0.9),
  ylim = c(0.95, 7),
  vertical = TRUE
)
ggsave("graphics/Fig1b_bz_y_mspe.pdf", y_b + y_c,
       width = 4.16, height = 2.4, units = "in",
       dpi = 600, device = cairo_pdf)

# MSPE(fzhat): UNORTH (left, viridis) | ORTH (right, inferno).
# Methods that emit constant 0 (base / posthoc_web) are greyed out.
fz_b <- make_plot(
  df_bz %>% filter(effect == "fz", metric == "mspe", model %in% UNORTH),
  ylab = TeX("$MSPE(\\hat{f}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  estimators = "DNN with Controls",
  show_legend = TRUE,
  strip_labels = FALSE
)
fz_c <- make_plot(
  df_bz %>% filter(effect == "fz", metric == "mspe", model %in% ORTH),
  ylab = TeX("$MSPE(\\hat{f}_Z)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  estimators = "DNN with Controls\n+ Orthogonalisation",
  show_legend = TRUE,
  strip_labels = FALSE
)
ggsave("graphics/Fig1b_bz_fz_mspe.pdf", fz_b + fz_c,
       width = 4.16, height = 2.4, units = "in",
       dpi = 600, device = cairo_pdf)


