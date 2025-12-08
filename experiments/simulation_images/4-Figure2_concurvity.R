library(readr)
library(dplyr)
library(ggplot2)
library(tikzDevice)
library(latex2exp)
library(png)
library(grid)
library(patchwork)

effect_names <- c(
  'y' = 'y',
  'fx' = 'Image Effect',
  'fr' = 'Residual Image Effect',
  'fz' = 'Covariate Effect'
)

df_bz <- read_csv("results/simulation_images/concurvity.csv") %>%
  mutate(model = factor(
    model, 
    levels= c("covar", "posthoc_lam0", "posthoc"),
    labels = c(
      "SGD",
      "Refit",
      "Pen. Refit")
  )) %>%
  filter(n < 50000) %>%
  mutate(effect = factor(effect, levels=c('y', 'fx', 'fr', 'fz')))



# Shared theme
shared_theme <- theme_bw() +
  theme(
    legend.title = element_text(vjust=-1.5),
    legend.title.position = "top",
    legend.key.height = unit(0.06, 'in'),
    legend.key.width = unit(0.15, 'in'),
    legend.ticks.length = unit(c(-.05, 0), 'in'),
    legend.ticks = element_line(color='black'),
    legend.position = c(0.4, 0.3),
    legend.direction = "vertical",
    #legend.margin = margin(-10, 0, 0, 0),   
    legend.background = element_rect(color = NA, fill = NA),
    legend.text = element_text(size = 7, family = "serif"),
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
    aes(x = n * 0.5, y = value, group = model)
  ) +
    geom_line(aes(color = model), alpha = 0.8, linewidth = 0.8) +
    geom_point(aes(color = model), alpha = 0.8, size = 0.8) +
    scale_x_log10(
      name = TeX("$N_{train}$ ($\\log_{10}$ scale)"),
      breaks = 100 * 2^(0:7)
    ) +
    scale_y_log10(name = ylab) +
    scale_color_viridis_d(name = "Method", option=color_option) +
    coord_cartesian(
      xlim = c(100, 100 * 2^7.5),
      ylim = c(10, .4 * 1e-4)
    ) +
    shared_theme
  
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
    model %in% c("SGD",
                 "Refit",
                 "Pen. Refit")
  ),
  ylab = TeX("$Bias^2(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = FALSE,
  strip_labels = FALSE
)

b2 <- make_plot(
  df_bz %>% filter(
    effect == "fx",
    metric == "var",
    model %in% c("SGD",
                 "Refit",
                 "Pen. Refit")
  ),
  ylab = TeX("$Var(\\hat{f}_X)$  ($\\log_{10}$ scale)"),
  color_option = "viridis",
  show_legend = TRUE,
  strip_labels = FALSE
)

b <- b1 + b2
plot(b)
#+ plot_annotation(
#  title = "b. Estimation of the direct X-effect", 
#  theme = theme(plot.title = element_text(size = 9, family = "serif"))
#  )

ggsave("graphics/Fig3_Concurvity.pdf", b, width = 3.5, height = 1.8, units = "in", dpi=600, device = cairo_pdf)

