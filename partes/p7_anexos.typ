#[
#heading(numbering: none)[Anexos]

#let anexo(body) = {
  set heading(numbering: "A", supplement: [Anexo])
  counter(heading).update(0)
  body
}

#show: anexo

#heading[Repositorio del Proyecto]
#link("https://github.com/daniel-alegria3/Proyecto_ML")

#heading[Fuente del Dataset]
#link("https://www.datosabiertos.gob.pe/dataset/dataset-de-la-estaci%C3%B3n-torre-de-gradiente-para-estimar-los-flujos-de-calor-sensible-y-calor")

]

