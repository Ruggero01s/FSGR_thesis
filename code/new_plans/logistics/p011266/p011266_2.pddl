(define (problem logistics_p011266_2)
(:domain logistics)
(:objects
	apn8 - airplane
	apt4 apt3 - airport
	tru2 tru1 - truck
	obj55 obj23 obj11 obj13 obj22 obj88 - package
	pos55 pos66 pos44 pos33 pos23 - location
)
(:init
	at tru1 pos66
	at tru2 pos44
	at obj55 pos23
	at obj23 pos55
	at obj88 pos44
	at obj11 pos23
	at obj22 pos23
	at obj13 pos66
	at apn8 apt4
)
(:goal (and
	at obj55 pos66
	at obj23 pos33
))
)