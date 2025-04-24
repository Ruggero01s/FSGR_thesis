(define (problem logistics_p011266_2)
(:domain logistics)
(:objects
	pos33 pos66 pos23 pos44 pos55 - location
	apn8 - airplane
	apt4 apt3 - airport
	tru2 tru1 - truck
	obj23 obj13 obj11 obj88 obj22 obj55 - package
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
	at obj11 pos33
	at obj13 pos23
	at obj11 pos66
	at obj55 pos55
))
)