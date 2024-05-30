library(tidyverse)
library(scales)

df = (
  list.files("~/research/picard-results/hopper/", full.names = TRUE)
  %>% map_df(read_csv)
)

(
  df %>% group_by(T, picard_tol) %>% summarise_all(funs(mean))
  %>% select(T, picard_tol, seq_time, picard_time)
  %>% gather(var, val, -T, -picard_tol)
  %>% ggplot(aes(T, val, color=factor(picard_tol)))
  + theme_minimal()
  + theme(legend.position="bottom")
  + geom_point()
  + geom_line()
  + facet_wrap(~ var, scales="free", label=label_both)
)

df %>% ggplot(aes(T, picard_tol, fill=picard_time)) + geom_()

(
  x
  %>% select(seq_time, picard_time)
  %>% gather(var, val)
  %>% ggplot(aes(val, color=var))
  + theme_minimal()
  + theme(legend.position="bottom")
  + stat_ecdf()
  + scale_y_continuous(label=percent, name="Cum %")
)


x <- read_csv("test.csv")

(
  x
  %>% group_by(seed)
  %>% mutate(first_error=first(obs_error))
  %>% ungroup
  ## %>% group_by(iteration)
  ## %>% summarise(action_error=mean(action_error))
  %>% ggplot(aes(iteration, action_error, color=factor(seed)))
  + theme_minimal()
  + theme(legend.position="bottom")
  + geom_line()
  + scale_y_log10()
)


(
  x
  %>% group_by(seed)
  %>% mutate(first_error=first(obs_error_final))
  %>% ungroup
  %>% group_by(iteration)
  %>% summarise(action_error=mean(action_error_final / first_error))
  %>% ggplot(aes(iteration, action_error))
  + theme_minimal()
  + theme(legend.position="bottom")
  + geom_line()
)



(
  x
  %>% group_by(seed)
  %>% arrange(iteration)
  %>% mutate(min_error=cummin(obs_error) / obs_norm)
  %>% ggplot(aes(iteration, min_error, color=factor(seed)))
  + theme_minimal()
  + theme(legend.position="bottom")
  + geom_line()
  + scale_y_log10()
  + geom_point()
)


(
  x
  %>% ggplot(aes(iteration))
  + theme_minimal()
  + theme(legend.position="bottom")
)


library(tidyverse)
library(scales)

envs = c(
    "hopper", "walker2d", "inverted_pendulum", "inverted_double_pendulum",
    "ant", "halfcheetah", "humanoid", "humanoid_standup", "reacher"
)

seeds= 0:29

## envs = c(
##     "hopper", "walker2d"
## )

df = (
  expand_grid(env=envs, seed=seeds)
  %>% mutate(path=file.path("results", env, seed, "timing.csv"))
  %>% `$`("path")
  %>% keep(file.exists)
  %>% map_df(read_csv)
)


(
  df
  %>% ggplot(aes(iteration, obs_error / obs_norm, color=factor(seed)))
  + theme_minimal()
  + theme(legend.position="none", )
  + geom_line(alpha=0.2)
  + geom_point(alpha=0.2)
  + facet_wrap(~ env_name, scales="free", label=label_both)
  + scale_y_log10()
)


(
  df
  %>% ggplot(aes(iteration, obs_error / obs_norm, color=factor(seed)))
  + theme_minimal()
  + theme(legend.position="none", )
  + geom_line(alpha=0.2)
  + geom_point(alpha=0.2)
  + facet_wrap(~ env_name, scales="free", label=label_both)
  + scale_y_log10()
)

# Final Plot
(
  df
  ## %>% filter(
  ##       env_name %in% c("hopper", "walker2d", "ant", "humanoid", "reacher"),
  ##     )
  %>% group_by(env_name, seed)
  %>% arrange(iteration)
  %>% mutate(first_error=first(obs_error_final), min_error=0)# =min(obs_error))
  %>% mutate(normed_error=(obs_error_final - min_error) / (first_error - min_error))
  # %>% mutate(normed_error=obs_error_final)
  %>% group_by(env_name, iteration)
  %>% summarise(
        mean=mean(normed_error),
        p50=quantile(normed_error, 0.5)[[1]],
        p20=quantile(normed_error, 0.2)[[1]],
        p80=quantile(normed_error, 0.8)[[1]],
        )
  %>% ggplot(aes(iteration, p50, ymin=p20, ymax=p80,
                 fill=env_name, color=env_name))
  + theme_minimal()
  + geom_line()
  + geom_point(alpha=0.2)
  ## + geom_point(alpha=0.2)
  + geom_ribbon(aes(color=NULL), alpha=0.1)
  # + facet_wrap(~ env_name, scales="free", label=label_both)
  + scale_x_continuous(name="Picard Iterations")
 + scale_y_log10(
     name="Normalized Rel. RMSE",
     breaks=c(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100),
   )
  + labs(fill="Environment", color="Environment")
  + coord_cartesian(xlim=c(0, 80), ylim=c(1e-6, 1))
)



tol = 1e-4
summ = (
  df
  %>% select(env_name, iteration, seed, ends_with("error"))
  %>% gather(var, error, -env_name, -iteration, -seed)
  %>% group_by(env_name, seed, var)
  %>% summarise(
        converged=max(error / obs_norm < tol),
        iters_to_converge=min(iteration[error / obs_norm < tol]),
        )
)
(
  summ
  %>% filter(var == "obs_error")
  %>% group_by(env_name, var)
  %>% arrange(iters_to_converge)
  %>% mutate(pct_converged=seq_along(iters_to_converge) / n())
  %>% ggplot(aes(iters_to_converge, pct_converged, color=factor(env_name)))
  + theme_minimal()
  + theme(legend.position="bottom")
  + stat_ecdf()
  + scale_y_continuous(label=percent, name="% Trajectories Converged")
  + scale_x_continuous(name="Picard iterations")
  + facet_wrap(~ var, scales="free", label=label_both)
)


(
  summ
  %>% filter(!(env_name %in% c("hopper", "halfcheetah")))
  %>% ggplot(aes(env_name, iters_to_converge))
  + geom_boxplot()
  + theme_minimal()
  + theme(legend.position="bottom")
)
