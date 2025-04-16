(define (problem logistics_p046760_3)
(:domain logistics)
(:objects
	apn7 apn2 apn6 apn3 apn1 apn5 - airplane
	apt6 apt8 apt7 apt5 - airport
	tru3 tru2 tru1 tru4 tru5 - truck
	obj66 obj55 obj13 obj11 obj33 obj23 obj00 obj44 obj88 obj12 - package
	pos13 pos11 pos21 pos22 pos55 pos66 pos44 pos33 pos23 - location
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
	at obj11 pos13
	at obj12 pos21
))
)