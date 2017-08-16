## Twiddle Algorithm

Twiddle algorithm shows an idea to tune parameters. Its result approximates a local optimal value.

```
epsilon = 0.001
p  = [p1_0, ..., pk_0] // initial value
dp = [dp1_0, ..., dpk_0] // initial tuning scale
best_err = run(p)
while (sum(dp) >= epsilon):
    for i in range(k):
        p[i] += dp[i]
        err = run(p)
        if err < best_err:
            best_err = err
            dp[i] *= 1.1
        else:
            p[i] -= 2*dp[i]
            err = run(p)
            if err < best_err:
                best_err = err
                dp[i] *= 1.1
            else:
                p[i] += dp[i]
                dp[i] *= 0.9
```

## Disadvantages
1. Local optimal value

Solution: Random choice of initial value

2. The algorithm choose the first drop of *best_err*, not the biggest.

## Template

[CarND-PID-Control-Project](https://github.com/fangchun007/CarND-PID-Control-Project/blob/master/src/main_steer.cpp)
