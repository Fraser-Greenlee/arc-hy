(import typing [Any Optional List Tuple Callable Annotated])
(import itertools [groupby])
(import functools [partial reduce])
(require hyrule * :readers *)
(import hyrule [*])
(import hyrule.collections *)

(defn last [sequence]
  "Returns the last element of a sequence"
  (get sequence -1))

(defn first [sequence]
  "Returns the first element of a sequence"
  (get sequence 0))

(defn rest [sequence]
  "Returns all elements after the first element"
  (cut sequence 1 None))

(defn butlast [sequence]
  "Returns all elements except the last one"
  (cut sequence 0 -1))

(defn take [n sequence]
  "Returns the first n elements of a sequence"
  (cut sequence 0 n))

(defn drop [n sequence]
  "Returns the sequence without its first n elements"
  (cut sequence n None))

(defn drop-last [n sequence]
  "Returns the sequence without its last n elements"
  (cut sequence 0 (- 0 n)))

(defn nth [sequence n [default None]]
  "Returns the nth element of a sequence, with optional default value"
  (try
    (get sequence n)
    (except [IndexError]
      default)))

(defn get-in [#^ Any data #^ list path]
  (if path
    (get-in (get data (first path)) (rest path))
    data))

(defn split-at [n sequence]
  "Splits a sequence at index n, returns a tuple of two sequences"
  [(take n sequence) (drop n sequence)])

(defn partition [n sequence [step None] [pad None]]
  "Returns a sequence of lists of n items each, at offsets step apart.
   If pad is provided, use it to fill incomplete partitions."
  (setv result [] step (if (is step None) n step))
  (for [i (range 0 (len sequence) step)]
    (setv chunk (take n (drop i sequence)))
    (when (and pad (< (len chunk) n))
      (setv chunk (+ chunk (* [pad] (- n (len chunk))))))
    (when (= (len chunk) n)
      (.append result chunk)))
  result)

(defn interleave [#* sequences]
  "Returns a sequence of the first item in each seq, then the second, etc."
  (if (not sequences)
    []
    (list (sum (zip #* sequences) []))))

(defn flatten [sequence]
  "Flattens a nested sequence one level deep"
  (sum sequence []))

(defn deep-flatten [sequence]
  "Completely flattens a nested sequence"
  (defn _flatten [s acc]
    (for [item s]
      (if (isinstance item (type []))
        (_flatten item acc)
        (.append acc item))))
  (setv result [])
  (_flatten sequence result)
  result)

(defn distinct [sequence]
  "Returns a sequence of unique elements, preserving order"
  (list (dict.fromkeys sequence)))

(defn frequencies [sequence]
  "Returns a dictionary of the number of occurrences of each item"
  (let [freq {}]
    (for [item sequence]
      (assoc freq item (+ (.get freq item 0) 1)))
    freq))

(defn group-by [f sequence]
  "Returns a dictionary of items grouped by (f item)"
  (setv groups {})
  (for [item sequence]
    (setv key (f item))
    (if (in key groups)
      (.append (get groups key) item)
      (assoc groups key [item])))
  groups)

(defn second [sequence]
  "Returns the second element of a sequence"
  (first (rest sequence)))

(defn partition-by [f coll]
  "copy of Clojure's partition-by function"
  (when (not coll) 
    (return []))
  (list (map list 
             (map second 
                 (groupby coll :key f)))))

(defn delimit-when [f coll]
  "split a sequence into sub-sequences using a function"
  (when (not coll) 
    (return []))
  (list (map list
    (map (fn [group] (list (map (fn [i] (get coll i)) group)))
      (map second 
        (groupby (range (len coll))  (fn [i] (sum (map f (cut coll 0 i))))))))))

(defn sliding [n sequence]
  "Returns a sequence of overlapping sub-sequences of length n"
  (if (>= n (len sequence))
    [(list sequence)]
    (list (map list (zip #* (list (map (fn [i] (drop i sequence))
                                      (range n))))))))

(defn reapply [f x n]
  "Reapply a function to an input several times"
  (if (<= n 0) x (reduce (fn [acc _] (f acc)) (range n) x)))