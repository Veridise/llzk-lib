(ns validate
  (:require ["node:fs/promises" :as fs]
            [promesa.core :as p]
            ["@mdx-js/mdx" :as mdx]))

(defn -main [file & _]
  (p/let [contents (fs/readFile file)
          _        (mdx/compile contents)]
    (println "Valid MDX")))
