(define (problem logistics_p019116_0)
(:domain logistics)
(:objects
	apn1 - airplane
	apt6 - airport
	tru5 tru1 - truck
	obj66 obj77 obj33 obj99 obj00 obj44 - package
	pos21 pos22 pos55 pos66 pos23 - location
)
(:init
	at tru5 pos22
	at tru1 pos55
	at obj00 pos23
	at obj44 pos66
	at obj77 pos21
	at obj66 pos66
	at obj33 pos22
	at obj99 pos55
	at apn1 apt6
)
(:goal (and
	at obj33 pos23
	at obj66 pos21
))
)