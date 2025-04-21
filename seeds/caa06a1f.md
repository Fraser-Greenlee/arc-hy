# Concepts

translational symmetry, symmetry detection, non-black background

# Description

The input defines a texture that does not extend to cover the entire canvas. To make the output, extend the texture to cover the entire canvas while shifting the texture left by one pixel.

## Input Representations

### Input 1

```hy
(->
  (new-grid
    (.array jnp [11 11]))

  (draw
    (.array jnp [0 0])
    (apply-texture
      (.array jnp
        [[Color.PINK Color.ORANGE]
         [Color.ORANGE Color.PINK]])
      (rect-mask
        (.array jnp [7 7]))))

  (color-background Color.GREEN))
```

### Input 2

```hy
(->
  (new-grid
    (.array jnp [8 8]))

  (draw
    (.array jnp [0 0])
    (apply-texture
      (.array jnp
        [[Color.PINK Color.GREEN]
         [Color.GREEN Color.PINK]])
      (rect-mask
        (.array jnp [7 7]))))

  (color-background Color.GREEN))
```

### Input 3

```hy
(->
  (new-grid
    (.array jnp [6 6]))

  (draw
    (.array jnp [0 0])
    (apply-texture
      (.array jnp
        [[Color.GREY Color.YELLOW]
         [Color.YELLOW Color.GREY]])
      (rect-mask
        (.array jnp [5 5]))))

  (color-background Color.PINK))
```

### Input 4

```hy
(->
  (new-grid
    (.array jnp [18 18]))

  (draw
    (.array jnp [0 0])
    (apply-texture
      (.array jnp
        [[Color.TEAL Color.GREY Color.ORANGE]
         [Color.GREY Color.ORANGE Color.TEAL]])
      (rect-mask
        (.array jnp [12 12]))))

  (color-background Color.GREEN))
```

## To Output Function

```python
def to_output(expr: hy.models.Expression) -> hy.models.Expression:
    grid_shape = hy_eval(expr[1][1])
    texture = hy_eval(code[2][2][1])
    return nested_list_as_hy_expr(
        ["->",
            ["new-grid", grid_shape],

            ["draw",
                [".array", "jnp", [0, 0]],
                ["apply-texture",
                    jnp.roll(texture, -1)
                    ["rect-mask" grid_shape]]]]
    )
```
