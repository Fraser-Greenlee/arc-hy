(require hyrule * :readers *)
(import hyrule [*])
(import hyrule.collections *)
(import enum [Enum IntEnum]
        typing [Any Optional List Tuple Callable Annotated]
        functools [partial reduce]
        operator
        hy.pyops [+ - * / //]
        hy [eval repr read]
        pydantic [BaseModel Field]
        jax.numpy :as jnp
        jax [Array]
        PIL [Image ImageDraw])
(import random [shuffle])
(import sequence *)
(import helpers *)


(setv MAX-GRID-DIM 30) ; Largest possible grid size is 30x30

(setv 2D (get Annotated #(Array "int32[* *]")) ; 2D is a jnp int array containing color values
      2Db (get Annotated #(Array "bool[* *]")) ; 2Db is a jnp bool array often used as a mask
      V (get Annotated #(Array "int32[2]")) ; V is a 2D vector with 2 int values often used to mark positions and sizes
      Args (get Tuple #(Any ...))) ; Args is a list of arbitrary arguments

(defclass Color [IntEnum]
    "
    Enum for colors

    Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON

    Use Color.ALL_COLORS for `set` of all possible colors (does not include UNSET)
    Use Color.NOT_BLACK for the same `set` without BLACK

    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing on them.
    (The exception is Color.UNSET, which is 0)
    "

  (setv UNSET 0
        BLACK 1
        BLUE 2
        RED 3 
        GREEN 4
        YELLOW 5
        GREY 6
        GRAY 6
        PINK 7
        ORANGE 8
        TEAL 9
        MAROON 10
        TRANSPARENT 1 ; sometimes the language model likes to pretend that there is something called transparent/background, and black is a reasonable default
        BACKGROUND 1)

  (defn #^ str __repr__ [self]
    (+ "Color." (. self name))))

; Define the color lists as separate class attributes after the enum definition
(setv Color.ALL_COLORS [Color.BLACK Color.BLUE Color.RED Color.GREEN Color.YELLOW Color.GREY Color.PINK Color.ORANGE Color.TEAL Color.MAROON]
      Color.NOT_BLACK [Color.BLUE Color.RED Color.GREEN Color.YELLOW Color.GREY Color.PINK Color.ORANGE Color.TEAL Color.MAROON])

(defn #^ 2D new-grid [#^ V size]
  "Create a 2D grid with unset values (0).

  Example usage:
  (new-grid [3 4])"
  (jnp.zeros size :dtype int))

(defn #^ 2D empty [#^ 2D grid]
  "Create an empty version of a 2D grid.

  Example usage:
  (empty arr)"
  (jnp.zeros_like grid :dtype int))

(defn #^ 2D color-background [#^ 2D grid #^ Color bg-color]
  "Color in the background of a grid.
  This means grid cells with an UNSET color will be replaced by the background color.

  Example usage:
  (color-background grid Color.BLUE)"
  (jnp.where (= grid Color.UNSET) bg-color grid))

(defn #^ 2D color-in [#^ Color color #^ 2Db mask]
  "Take a mask (boolean 2D array) and set it to a given color.

  Example usage:
  (color-in Color.GREEN (point-mask))"
  (* mask color))

(defn #^ (get Callable #([2D Color Args] 2D)) color-overlap [#^ Color overlap-color #^ (get Callable #([2D Args] 2D)) draw-fn]
  "Apply an overlap color to a given draw function.
  This means that non UNSET colors already on the grid will be filled with the overlap color when drawn.

  Example usage:
  ((color-overlap Color.YELLOW draw) grid (jnp.array [0 0]) (color-in Color.TEAL (rect-mask (jnp.array [2 3]))))"
  (fn [grid #* args]
    (jnp.where
      (&
        (!= grid Color.UNSET)
        (!= (draw-fn (empty grid) #* args) Color.UNSET))
      overlap-color
      (draw-fn grid #* args))))

(defn #^ 2D apply-texture [#^ 2D texture #^ 2Db mask]
  "Apply a colored texture to a mask.
  The texture always starts from the top-left corner of the mask.

  Example usage:
  (apply-texture (jnp.array [[Color.PINK Color.ORANGE] [Color.ORANGE Color.PINK]]) (rect-mask (jnp.array [7 7])))"
  (let [rows (jnp.arange (get (. mask shape) 0))
        cols (jnp.arange (get (. mask shape) 1))
        row-indices (% (.reshape rows #(-1 1)) (get (. texture shape) 0))
        col-indices (% (.reshape cols #(1 -1)) (get (. texture shape) 1))]
    (* 
      (get texture #(row-indices col-indices))
      (get mask #(row-indices col-indices)))))

(defn #^ 2D draw [#^ 2D grid #^ V position #^ 2D shape]
  "Draw a given shape on the grid at a position.
  The shape will be drawn with its top-left at position on the grid.

  Example usage:
  (draw (jnp.array [3 3]) (color-in Color.RED (point-mask)))"
  (let [start (jnp.maximum position 0)
        end (jnp.minimum (+ position (jnp.array shape.shape)) (jnp.array grid.shape))
        grid-slice #(
          (slice (get start 0) (get end 0))
          (slice (get start 1) (get end 1))
        )
        shape-start (- start position)
        shape-end (- end position)
        shape-slice #(
          (slice (get shape-start 0) (get shape-end 0))
          (slice (get shape-start 1) (get shape-end 1))
        )]
    (
      (. (get grid.at grid-slice) set)
      (jnp.where
        (!= (get shape shape-slice) 0)
        (get shape shape-slice)
        (get grid grid-slice))
    )))

(defn #^ 2D draw-centre [#^ 2D grid #^ V pos-centre #^ 2D shape]
  "Draw a given shape on the grid with the shapes centre at pos-centre.

  Example usage:
  (draw-centre grid (jnp.array [4 3]) (color-in Color.ORANGE (plus-mask 18)))"
  (draw grid (- pos-centre (// (jnp.array (. shape shape)) 2)) shape))

(defn #^ bool on-grid [#^ 2D grid #^ V pos]
  "Check if a position is on the grid.

  Example usage:
  (grid (jnp.array [13 53]))"
  (and
    (>= (get pos 0) 0)
    (>= (get pos 1) 0)
    (< (get pos 0) (get (. grid shape) 0))
    (< (get pos 1) (get (. grid shape) 1))))

(defn #^ 2Db -flood-fill-mask-step [#^ 2Db grid-mask #^ V position]
  (let [step-to-pos #%(let [shift-pos (- position (jnp.array %2))] (if (and (on-grid %1 shift-pos) (= (get %1 (tuple shift-pos)) 1)) (flood-fill-mask-step %1 shift-pos) %1))]
    (-> (.set (get (. grid-mask at) (tuple position)) 2) (step-to-pos [1 0]) (step-to-pos [0 1]) (step-to-pos [0 -1]) (step-to-pos [-1 0]))))

(defn #^ 2Db flood-fill-mask [#^ 2D grid #^ V position]
  "Fill the connected region on the grid that contains the point at position and all have the same color.

  Example usage:
  (flood-fill-mask (bit-map maze) (jnp.array [7 17]))"
  (let [fill-color (.item (get grid (tuple position)))]
    (= (-flood-fill-mask-step (.astype (= grid fill-color) int) position) 2)))

(defn #^ 2D thicken [#^ V thickness #^ 2D shape]
  "Thicken a shape in both the y and x directions by a given thickness vector.

  Example usage:
  (thicken (jnp.array [2 2]) (bit-map shared-mask))"
  (-> shape
      (jnp.repeat (get thickness 0) :axis 0)
      (jnp.repeat (get thickness 1) :axis 1)))

(defn #^ 2D add-lr-mirror [#^ 2D arr]
  "Concatenate the left-to-right mirror image of a 2D array on the x-plane."
  (jnp.concatenate [arr (jnp.fliplr arr)] :axis 1))

(defn #^ 2D add-ud-mirror [#^ 2D arr]
  "Concatenate the up-to-down mirror image of a 2D array on the y-plane."
  (jnp.concatenate [arr (jnp.flipud arr)]))

(defn #^ 2D add-quad-mirror [#^ 2D shape]
  "Create a larger shape with both x and y plane mirror symmetry by applying both left-right and up-down mirroring.
  Takes a 2D array and returns a new array with the input mirrored in both x and y planes,
  effectively creating 4 copies of the original array arranged in quadrants."
  (-> shape
      (add-lr-mirror)
      (add-ud-mirror)))

(defn #^ 2D add-quad-rotate [#^ 2D shape #^ int pad-size #^ Color pad-color]
  "Create a larger shape with 4-way rotational symmetry by concatenating 90 degree rotations to the shape.
  The rotations are padded with a pad-color creating a plus shape between the rotations.
  Set pad-size to 0 for no padding."
  (jnp.concatenate [
    (jnp.concatenate [
      shape
      (* (jnp.ones [pad-size (get (. shape shape) 1)]) pad-color)
      (reapply jnp.rot90 shape 1)
    ])
    (* (jnp.ones [(+ (* (get (. shape shape) 1) 2) pad-size) pad-size]) pad-color)
    (jnp.concatenate [
      (reapply jnp.rot90 shape 3)
      (* (jnp.ones [pad-size (get (. shape shape) 1)]) pad-color)
      (reapply jnp.rot90 shape 2)
    ])
  ] :axis 1))

(defn #^ 2Db line-mask [#^ V delta]
  "Create a simple mask which is just a line with direction `delta`."
  (let [delta (ncut delta ::-1)
        img (Image.new "1" (tuple (+ (abs delta) 1)) 0)]
    ((. (ImageDraw.Draw img) line) [#(0 0) (tuple (abs delta))] :fill 1)
    (as-> (jnp.array img :dtype bool) arr
        (if (< (get delta 0) 0) (jnp.flipud arr) arr)
        (if (< (get delta 1) 0) (jnp.fliplr arr) arr))))

(defn #^ 2Db rect-mask [#^ V size]
  "Create a simple rectangle mask with size `size`."
  (jnp.full size True))

(defn #^ 2Db frame-mask [#^ V size #^ int thickness]
  "Create a frame mask, as in a hollow rectangle with a given thickness."
  (let [add-side #%((. (get %1.at %2) set) True)]
    ( -> (jnp.full size False)
         (add-side (slice None thickness))
         (add-side #((slice None None) (slice None thickness)))
         (add-side #((slice None None) (slice (- thickness) None) None))
         (add-side (slice (- thickness) None)))))

(defn #^ 2Db pyramid-mask [#^ int size]
  "Create a pyramid-shaped mask using triangular matrices.
  Returns a 2D boolean array where the mask forms a symmetric triangle pattern
  with one flat side, resembling a side view of a pyramid."
  (jnp.concatenate [(ncut (jnp.tri size) : ::-1) (ncut (jnp.tri size) : 1:)] :axis 1))

(defn #^ 2Db point-mask []
  "Create a mask that is just a single point."
  (jnp.full [1 1] True))

(defn #^ 2Db plus-mask [#^ V size #^ int [thickness 1]]
  "Create a plus shaped mask.
  Size indicates the length of each of the lines extending from its centre point (so size=2 has a height of 2+1+2=5)
  Thickness indicates the thickness of the plus lines."
  (.astype (add-quad-rotate (new-grid size) thickness 1) bool))

(defn #^ 2Db bit-map [#^ 2D values]
  "Create a mask from a 2D list of boolean values.
  Used to represent arbitrary patterns."
  (jnp.array values :dtype bool))

(defn #^ 2Db crop [#^ V size #^ 2D shape]
  "Crop a shape from the top-left with a rectangle of size.

  Example usage:
  (crop 3 (jnp.array [1 1]) (cross-mask (jnp.array [1 1]))) ; results in a 3x3 cross mask"
  (get shape #((slice None (get size 0)) (slice None (get size 1)))))

(defn #^ 2Db cross-mask [#^ V centre]
  "Create a cross shaped mask.
  This shape is unique in that only its centre is specified and the resulting shape is beyond the max grid size.
  This means that the cross-mask will always fill its container."
  (bit-map (draw (new-grid [30 30]) (- centre 30) 
    (* (| (line-mask (jnp.array [60 60])) (line-mask (jnp.array [60 -60]))) 1)
  )))

; TODO (at some point) best-fit
; - modifies the shape so far to match the pixels on the grid
; - helpful for shapes that are skwed or noised in some way
; - odd shape is then taken from the input and can be used in the output
; - for task generation the best-fit inputs are random
; - for task solving time they are taken from the real grid
