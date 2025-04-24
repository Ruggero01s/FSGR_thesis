;;(;metadata (recognizability:0.18)
(define (problem logistics_p046760_0)
(:domain logistics)
(:objects
	pos21 pos33 pos66 pos22 pos11 pos13 pos23 pos44 pos55 - location
	apn1 apn7 apn3 apn2 apn5 apn6 - airplane
	apt7 apt6 apt8 apt5 - airport
	tru4 tru2 tru5 tru3 tru1 - truck
	obj33 obj11 obj12 obj13 obj23 obj44 obj88 obj66 obj00 obj55 - package
	)
(:init
	at tru2 pos44
	at tru4 pos22
	at tru1 pos66
	at tru5 pos23
	at tru3 pos21
	at obj66 pos33
	at obj13 pos13
	at obj11 pos11
	at obj12 pos11
	at obj00 pos44
	at obj23 pos33
	at obj44 pos66
	at obj55 pos21
	at obj88 pos55
	at obj33 pos44
	at apn6 apt7
	at apn3 apt7
	at apn2 apt7
	at apn1 apt7
	at apn7 apt6
	at apn5 apt6
)
(:goal (and
	at obj11 pos33
	at obj00 pos21
	at obj88 pos13
	at obj44 pos55
))
)