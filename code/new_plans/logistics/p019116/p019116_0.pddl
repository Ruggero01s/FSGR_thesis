;;(;metadata (recognizability:0.2)
(define (problem logistics_p019116_0)
(:domain logistics)
(:objects
	pos21 pos66 pos22 pos23 pos55 - location
	apn1 - airplane
	apt6 - airport
	tru5 tru1 - truck
	obj33 obj99 obj44 obj66 obj77 obj00 - package
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
	at obj66 pos22
	at obj99 pos55
	at obj44 pos66
	at obj77 pos55
))
)