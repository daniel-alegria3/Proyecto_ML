#import "./caratula_unsaac.typ": tarea

#show: tarea.with(
  course: [APRENDIZAJE AUTOMÁTICO],
  title: [Predicción de flujo de calor del suelo usando variables meteorológicas],
  professor: [Javier Arturo Rozas Huacho],
  date: "Junio 2025",
  authors: (
    (
      name: "Aguilar Mainicta Gian Marco",
      id: "174905",
    ),
    (
      name: "Alegria Sallo Daniel Rodrigo",
      id: "215270",
    ),
    (
      name: "Muñoz Centeno Milder",
      id: "211860",
    ),
  ),
)

//============================== {Configuraciones} =============================
#show outline.entry.where(
  level: 1
): set block(above: 1.4em)

#show heading.where(
  level: 1
): set block(below: 1em)

#set list(
  indent: 1.2em,
  spacing: 0.85em,
)

#set enum(
  indent: 1.2em,
  spacing: 0.85em,
)

//================================ {Documento} =================================

#outline(
  title: "Tabla de Contenido",
)
#pagebreak()

#include "partes/p1_introduccion_problema.typ"
#include "partes/p2_descripcion_dataset.typ"
#include "partes/p3_formulacion_problema.typ"
#include "partes/p4_metodologia_preprocesamiento.typ"
#include "partes/p5_modelo.typ"
#pagebreak()
#include "partes/p6_conclusiones.typ"
#pagebreak()
#include "partes/p7_anexos.typ"

