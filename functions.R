grid_arrange_shared_legend <-function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
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
    x=rep(1:ncol, times=ncol),
    y=rep(ncol:1, times=ncol),
    val=grid
  )
  
  if (levels) {
    data.frame$val <- factor(data.frame$val, levels = c("S", "F", "H", "G"),  labels = c("S", "F", "H", "G"),  ordered=TRUE)
  }
  return(data.frame)
}

grids <- list()
grids[[8]] = read.csv("out/grids-8.csv", header=FALSE, stringsAsFactors=FALSE, colClasses = c("character"))
grids[[4]] = read.csv("out/grids-4.csv", header=FALSE, stringsAsFactors=FALSE, colClasses = c("character"))
grids[[8]][,"V1"] <- as.numeric(grids[[8]][,"V1"])
grids[[4]][,"V1"] <- as.numeric(grids[[4]][,"V1"])

lakes <- list()
lakes[[4]] <- list()
lakes[[8]] <- list()

for (i in c(4, 8)) {
  for (id in 1:8) {
    # take the grid from i (4 or 8) grid and take the problem id "id"
    lake <- grids[[i]][id,-1]
    lake <- as.factor(lake)
    lakes[[i]][[id]] <- create_grid(lake, i)
  }
}

get_label <- function(x, y, val) {
  return(val)
  return(data.frame.grid[data.frame.grid$x == x & data.frame.grid$y == y, "val"])
  
  if (!is.null(val)) {
    return(val)
  }
  return("X")
}

finish_plot <- function(plot) {
  plot <- plot + geom_tile(aes(fill = val), colour = "white") + 
    geom_text(aes(label = get_label(x,y,val)), position = position_dodge(width=0.9), size=5) +
    scale_fill_manual(values=palette, labels = c("Start (safe)", "Frozen (safe)", "Hole", "Goal")) + 
    scale_x_discrete(expand=c(0,0)) +
    scale_y_discrete(expand=c(0,0)) +
    theme_void() +
    labs(fill="") +
    theme(legend.position="right", legend.direction = "horizontal", legend.box = "vertical")
  theme(legend.title = element_blank(), legend.margin=margin(t=1, unit="cm"), legend.text=element_text(size=23))
  return(plot)
}

