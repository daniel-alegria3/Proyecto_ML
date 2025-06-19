#let tarea(
  course: none,
  title: none,
  professor: none,
  date: none,
  authors: (),
  doc,
) = {
  let fontsize = 12pt

  let margin = 2.54cm
  let margin_side_ratio = 2%
  let caratula_margin_ratio = 2%

  let escudos_ratio = 70%

  // Temp fix for 'lang: es' not working on datetime()
  let months = ("Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre")

  //================================= {General} ==================================
  set page(
    paper: "a4",
    margin: (
      rest: margin,
      inside: margin + margin_side_ratio,
      outside: margin - margin_side_ratio,
    ),
    // header: align(right + horizon)[
    //   _ #title _
    // ],
    // numbering: "1",
    // columns: 2,
  )
  set text(
    size: fontsize,
    lang: "es",
  )
  set par(
    first-line-indent: 1.2em,
    spacing: 1.2em,
    leading: 0.65em,
    justify: true,
  )
  set heading(
    numbering: "1.",
  )
  set document()

  //================================ {Overrides} =================================
  // show heading: it => [
  //   #set align(center)
  //   #set text(13pt, weight: "regular")
  //   #block(smallcaps(it.body))
  // ]
  //
  // show heading.where(
  //   level: 2
  // ): it => text(
  //   size: 11pt,
  //   weight: "regular",
  //   style: "italic",
  //   it.body + [.],
  // )

  //================================= {Caratula} =================================
  page(
    margin: (
      rest: margin + caratula_margin_ratio,
    )
  )[
    #place(
      float: true,
      top + center,
      scope: "parent",
      clearance: 2em,
    )[
      #set par(first-line-indent: 0em)

      #text(fontsize*1.35)[
        UNIVERSIDAD NACIONAL DE SAN ANTONIO ABAD DEL CUSCO
      ]

      #text(fontsize*1.4)[
        FACULTAD DE INGENIERÍA ELÉCTRICA, ELECTRÓNICA, INFORMÁTICA Y MECÁNICA
      ]

      #text(fontsize*1.26)[
        ESCUELA PROFESIONAL DE INGENIERÍA INFORMÁTICA Y DE SISTEMAS
      ]

      #grid(
        columns: (1fr, 1fr),
        figure(
          image("./imagenes/unsaac_logo.png", width: escudos_ratio),
        ),
        figure(
          image("./imagenes/facultad_logo.png", width: escudos_ratio)
        )
      )

      #text(fontsize*1.35)[
        #course
      ]

      #v(1em)
      #text(fontsize*1.35)[
        *#title*
      ]

      #v(1.5em)
      #text(fontsize*1.1)[
        #align(left)[
          #set par(justify: false)

          #if professor != none [
            DOCENTE: #h(1fr) #professor
          ]

          #if authors.len() > 1 [
            INTEGRANTES:

            #grid(
              columns: (7fr, 1fr),
              row-gutter: 1em,
              align: right,

              ..authors.map(author => (
                author.name,
                [(#author.id)]
              )).flatten(),
            )
          ] else if authors.len() == 1 [
            #let (name, id) = authors.at(0)
            ALUMNO: #h(7fr) #name #h(1fr) (#id)
          ]
        ]
      ]


      #v(1fr)
      // # NOTE: bottom is not what puts it near foot of page
      #text(fontsize*1.2)[
        #align(center + bottom)[
          Perú \
          #if date != none {
            date
          } else {
            let today = datetime.today()
            [ #months.at(today.month()-1) del #today.year() ]
          }
        ]
      ]
    ]
  ]
  pagebreak()

  //============================== {Document Body} ===============================
  doc
}

