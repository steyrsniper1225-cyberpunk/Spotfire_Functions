{
    "Desc" : "Add_AutoMail_Memo",
    "Columns" : {
        "WINDOWFRAME" : ["MONTH", "-", "WEEK", "_", "DATE"],
        "WINDOW" : {
            "MONTH" : "YY-MM" & "M",
            "WEEK" : "YY-WW" & "W",
            "DATE" : "YY-MM/DD"
        },
        "LINE" : [
            "TOTAL",
            "PHT03", "PHT09", "PHT13", "PHT14",
            "OVN01", "OVN02", "OVN03", "OVN04", "OVN05", "OVN06", "OVN07", "OVN08"
            ],
        "GLASS" : int,
        "CODE" : ["PLN1-FM", "PLN1-SM", "PLN2-FM", "PLN2-SM"],
        "DPU" : float
    }
}
