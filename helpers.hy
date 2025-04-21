(require hyrule * :readers *)
(import enum [Enum IntEnum]
        functools [partial reduce]
        operator
        hy.pyops [+ - * / //]
        hy [eval]
        hy.models :as m
        pydantic [BaseModel Field]
        jax.numpy :as jnp
        jax [Array]
        PIL [Image ImageDraw])
(import random [choices])
(import sequence *)

(defn unq [x]
  "Required before unquoting enums like Color values."
  (read (repr x)))

(defn hystringify [value]
  (let [sv (str value)]
    (if (.startswith sv "is_")
      (+ (cut sv 3) "?")
      sv)))

; print a Hy form as its own code
(defn to_source [form]
  (cond
    (isinstance form m.Expression)
      (+ "(" (.join " " (map to_source form)) ")")
    (isinstance form m.Symbol)
        (hystringify form)
    (or (isinstance form m.Integer) (isinstance form int))
        (str (eval form))
    (isinstance form m.Keyword)
        (hystringify form)
    (or (isinstance form m.String) (isinstance form str))
        (str (+ "\"" (str (eval form)) "\""))
    (or (isinstance form m.Dict) (isinstance form dict))
        (+ "{" (.join " " (map to_source form)) "}")
    (or (isinstance form m.List) (isinstance form list))
        (+ "[" (.join " " (map to_source form)) "]")
    (or (isinstance form m.Tuple) (isinstance form tuple))
        (+ "#(" (.join " " (map to_source form)) ")")
    (or (isinstance form Array))
        (+ "((. jnp array) " (to_source (.tolist form)) ")")
    ; TODO: support Color?
    True
        (raise (NotImplementedError [(type form) form]))))

(defn rand-lofl [h w [options [0 1]]]
  (lfor _ (range h) (choices options :k w)))
