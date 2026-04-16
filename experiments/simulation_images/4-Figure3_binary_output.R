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

df_bz <- read_csv("results/simulation_images/binary_increasing_bz.csv") %>%
  mutate(model = factor(
    model, 
    levels= c( "posthoc", "posthoc_orth", "base", "posthoc_web"),
    labels = c(
      "DNN w. Controls",
      "DNN w. Controls + Orth.",
      "DNN",
      "DNN + Orth. [17]")
  )) %>%
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
  make_plot <- function(data, ylab, color_option, show_legend = TRUE, strip_labels = TRUE) {
    p <- ggplot(
      data,
      aes(x = n * 0.5, y = value, group = bz)
    ) +
      #geom_hline(yintercept = 0, linewidth = 0.5, color = "grey80") +
      geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
      geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
      scale_x_log10(
        name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
        breaks = 100 * 2^(0:7)
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
        xlim = c(100, 100 * 2^7.05),
        ylim = c(.4 * 1e-3, 1.25)
      ) +
      shared_theme +
      facet_grid(model ~ .)
    
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
    model %in% c("DNN", "DNN w. Controls")
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
    model %in% c("DNN w. Controls + Orth.", "DNN + Orth. [17]")
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
    model %in% c("DNN", "DNN w. Controls")
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
    model %in% c("DNN w. Controls + Orth.", "DNN + Orth. [17]")
  ),
  ylab = TeX("$Var(\\hat{f}^{re}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
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

ggsave("graphics/Fig3a_binary_bz_bias.pdf", b, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)
ggsave("graphics/Fig3b_binary_bz_var.pdf", c, width = 4.16, height = 2.8, units = "in", dpi=600, device = cairo_pdf)


# Shared theme 2
shared_theme2 <- theme_bw() +
  theme(
    legend.title.position = "left",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.3, 0.8),
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
make_plot2 <- function(data, ylab, color_option, show_legend = TRUE, strip_labels = TRUE) {
  p <- ggplot(
    data,
    aes(x = n * 0.5, y = value, group = bz)
  ) +
    #geom_hline(yintercept = 0, linewidth = 0.5, color = "grey80") +
    geom_line(aes(color = bz), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = bz), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:7)
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
      xlim = c(100, 100 * 2^7.05),
      ylim = c(.4 * 1e-3, 1.25)
    ) +
    shared_theme2 +
    facet_grid(. ~ model)
  
  if (!show_legend) {
    p <- p + theme(legend.position = "none")
  }
  if (strip_labels) {
    p <- p + theme(strip.text.y = element_blank())
  }
  
  p
}


# Build plots
a <- make_plot2(
  df_bz %>% filter(
    effect == "fx",
    metric == "mspe",
    model %in% c("DNN", "DNN w. Controls")
  ),
  ylab = TeX("$MSPE(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE
)

a

ggsave("graphics/Fig3_binary_bz.pdf", a, width = 3.5, height = 1.8, units = "in", dpi=600, device = cairo_pdf)


# Build plots
b <- make_plot2(
  df_bz %>% filter(
    effect == "fr",
    metric == "mspe",
    model %in% c("DNN w. Controls + Orth.", "DNN + Orth. [17]")
  ),
  ylab = TeX("$MSPE(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "inferno",
  show_legend = TRUE,
  strip_labels = FALSE
)

b

a/b

ggsave("graphics/Fig3_binary_bz_full.pdf", a/b, width = 3.5, height = 2.8, units = "in", dpi=600, device = cairo_pdf)


