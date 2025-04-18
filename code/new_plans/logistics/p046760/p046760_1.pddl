(define (problem logistics_p046760_1)
(:domain logistics)
(:objects
	pos33 pos66 pos11 pos13 pos23 pos22 pos21 pos44 pos55 - location
	apn5 apn3 apn2 apn6 apn1 apn7 - airplane
	apt6 apt8 apt7 apt5 - airport
	tru3 tru1 tru4 tru5 tru2 - truck
	obj44 obj11 obj88 obj00 obj23 obj13 obj66 obj55 obj33 obj12 - package
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
	at obj33 pos33
	at obj13 pos13
	at obj23 pos55
))
)