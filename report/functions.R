grid_arrange_shared_legend <-function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right", "top")) {
  plots <- list(...)
  position <- match.arg(position)
  g <-
    ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position = "none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
  
  combined <- switch(
    position,
    "top" = arrangeGrob(
      legend,
      do.call(arrangeGrob, gl),
      ncol = 1,
      heights = lheight, unit.c(unit(1, "npc") - lheight)
    ),
    "bottom" = arrangeGrob(
      do.call(arrangeGrob, gl),
      legend,
      ncol = 1,
      heights = unit.c(unit(1, "npc") - lheight, lheight)
    ),
    "right" = arrangeGrob(
      do.call(arrangeGrob, gl),
      legend,
      ncol = 2,
      widths = unit.c(unit(1, "npc") - lwidth, lwidth)
    )
  )
  
  grid.draw(combined)
}


#####
# This function creates a grid plot

create_grid <- function(grid, ncol, levels=TRUE) {
  data.frame <- data.frame(
    x_grid=rep(1:ncol, times=ncol),
    y_grid=rep(ncol:1, each=ncol),
    x=rep(0:(ncol-1), times=ncol),
    y=rep(0:(ncol-1), each=ncol),
    val=grid
  )
  
  if (levels) {
    data.frame$val <- factor(data.frame$val, levels = c("S", "F", "H", "G"),  labels = c("S", "F", "H", "G"),  ordered=TRUE)
  }
  return(data.frame)
}

get_label <- function(val, action, action_passive, key="val") {
  if (key == "policy") {
    return(action)
  }
  
  if (key == "passive_policy") {
    return(action_passive)
  }
  return(val)
}

u_plot <- function(data) {
  plot <- remove_axis(remove_legend(ggplot(data = data, mapping = aes(x = x_grid, y = y_grid, fill = u)) +
    geom_tile() +
    geom_text(aes(label = round(u, digits=3)), position = position_dodge(width=0.9), size=3) +
    scale_fill_gradient(low = "#1d4877", high = "#ee3e32") +
    theme_void()))
  
  return(plot)
}

env_plot <- function(plot, key="val") {
  plot <- plot + geom_tile(aes(fill = val), colour = "white") + 
    geom_text(aes(label = get_label(val, action_grid, action_passive_grid, key)), position = position_dodge(width=0.9), size=5) +
    scale_fill_manual(values=palette, labels = c("Start (safe)", "Frozen (safe)", "Hole", "Goal")) + 
    theme_void() +
    theme(legend.position="right", legend.direction = "horizontal", legend.box = "vertical")
  theme(legend.title = element_blank(), legend.margin=margin(t=1, unit="cm"), legend.text=element_text(size=23))
  return(remove_axis(plot))
}

remove_legend <- function(plot) {
  return(plot + theme(legend.title = element_blank()) + theme(legend.position = "none"))
}

remove_axis <- function(plot) {
  plot <- plot + scale_x_discrete(expand=c(0,0)) +
    scale_y_discrete(expand=c(0,0)) +
    labs(fill="")
  return(plot)
}

fix_caption <- function(plot, margin=c(0.2,0.2,0.2,0.2)) {
  return(plot + theme(plot.margin=unit(margin, "cm")) 
         + theme(plot.caption = element_text(hjust = 0.5, size = 14)))
}

arrow_from_cell <- function(cell, index="action") {
  if (cell["val"] %in% c("H", "G")) {
    return(cell["val"])
  }
  if (is.na(cell[index])) {
    return("NA")
  }
  if (cell[index] == "right") {
    return(">")
  }
  if (cell[index] == "up") {
    return("^")
  } 
  if (cell[index] == "down") {
    return("v")
  }   
  if (cell[index] == "left") {
    return("<")
  }
  return(cell["val"])
}

arrow_passive_from_cell <- function(cell) {
  return(arrow_from_cell(cell, "passive_action"))
}
