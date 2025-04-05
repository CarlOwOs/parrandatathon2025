import os
import json
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures
from typing import List, Tuple

def excluded_key(key,debug=False):
    excluded_endings = ['.css', '.pdf']
    for k in excluded_endings:
        if key.endswith(k):
            if debug:
                print("bad key")
            return True
    return False

def is_interesting(client,key,content, max_length=200):
    if excluded_key(key,debug=True):
        return "No"
        
    examples = [("https://www.1-act.com/thermal-solutions/enclosure-cooling/vapor-compression-coolers/","VCC (Vapor Compression Coolers) - Advanced Cooling Technologies\n|\nADVANCED COOLING TECHNOLOGIES\ncontact\nThermal Solutions\nPassive Thermal Solutions\nHeat Pipes\nHiK Plates­™/ Heat Pipe Assemblies\nPulsating Heat Pipes\nVapor Chambers\nHigh Temperature Heat Pipes\nLoop Thermosyphons\nPhase Change Based Solutions\nPCM Heat Sinks\nCustom Cold Plates\nActive Thermal Management Solutions\nLiquid Cooling\nPumped Two-Phase\nLiquid-Air HX\nTekgard® ECUs\nVaphtek™ ECU\nTekgard® Chillers\nEmbedded Computing Solutions\nICE-Lok®\nVME/ VPX Card Frames\nConduction Cooled Chassis\nLiquid Cooled Chassis\nEnclosure Cooling Products\nHSC (Heat Sink Coolers)\nHPC (Heat Pipe Coolers)\nTEC (Thermoeletric Coolers)\nVCC (Vapor Compression Coolers)\nEnclosure Cooling Selection Tool\nHVAC Energy Recovery\nAAHX\nWAHX\nSpace Thermal Control\nConstant Conductance Heat Pipes\nVariable Conductance Heat Pipes\nSpace Copper-Water Heat Pipes\nLoop Heat Pipes\nSpace VPX\nLiquid Cooling\nEngineering Services\nResearch & Development\nTeam & Value Add\nEmerging Technology\nTechnical Papers\nOther Research Interests\nProduct Development\nSpace Thermal & Structural Analysis\nManufacturing\nLifecycle Management\nIndustries\nEnergy\nSpace\nHVAC Energy Recovery\nDefense\nMedical\nData Centers\nOther\nResources\nBlog\nCalculators & Selection Tools\nAAHX Selection Tool\nEnclosure Cooling Selection Tool\nHeat Pipe Calculator\nPCM Calculator\nWAHX Selection Tool\nPublications\nPublished Articles\nPatents\nTechnical Papers\nLearning Center\nHeat Pipe Learning Center\nPumped Two-Phase Learning Center\nPCM Learning Center\nHVAC Learning Center\nVideos\neBooks\nBrochures\nCase Studies\nWebinars\nFind your Rep\nShop\nAbout\nCareers\nEvents\nNews\nSustainability\nACT Leadership\nContact\nThermal Solutions\nPassive Thermal Solutions\nHeat Pipes\nHiK Plates­™/ Heat Pipe Assemblies\nPulsating Heat Pipes\nVapor Chambers\nHigh Temperature Heat Pipes\nLoop Thermosyphons\nPhase Change Based Solutions\nPCM Heat Sinks\nCustom Cold Plates\nActive Thermal Management Solutions\nLiquid Cooling\nPumped Two-Phase\nLiquid-Air HX\nTekgard® ECUs\nVaphtek™ ECU\nTekgard® Chillers\nEmbedded Computing Solutions\nICE-Lok®\nVME/ VPX Card Frames\nConduction Cooled Chassis\nLiquid Cooled Chassis\nEnclosure Cooling Products\nHSC (Heat Sink Coolers)\nHPC (Heat Pipe Coolers)\nTEC (Thermoeletric Coolers)\nVCC (Vapor Compression Coolers)\nEnclosure Cooling Selection Tool\nHVAC Energy Recovery\nAAHX\nWAHX\nSpace Thermal Control\nConstant Conductance Heat Pipes\nVariable Conductance Heat Pipes\nSpace Copper-Water Heat Pipes\nLoop Heat Pipes\nSpace VPX\nLiquid Cooling\nEngineering Services\nResearch & Development\nTeam & Value Add\nEmerging Technology\nTechnical Papers\nOther Research Interests\nProduct Development\nSpace Thermal & Structural Analysis\nManufacturing\nLifecycle Management\nIndustries\nEnergy\nSpace\nHVAC Energy Recovery\nDefense\nMedical\nData Centers\nOther\nResources\nBlog\nCalculators & Selection Tools\nAAHX Selection Tool\nEnclosure Cooling Selection Tool\nHeat Pipe Calculator\nPCM Calculator\nWAHX Selection Tool\nPublications\nPublished Articles\nPatents\nTechnical Papers\nLearning Center\nHeat Pipe Learning Center\nPumped Two-Phase Learning Center\nPCM Learning Center\nHVAC Learning Center\nVideos\neBooks\nBrochures\nCase Studies\nWebinars\nFind your Rep\nShop\nAbout\nCareers\nEvents\nNews\nSustainability\nACT Leadership\nContact\nNEMA Ratings\nHSC (Heat Sink Coolers)\nHPC (Heat Pipe Coolers)\nTEC (Thermoeletric Coolers)\nVCC (Vapor Compression Coolers)\nThermal Solutions\n|\nEnclosure Cooling Products\n|\nVCC (Vapor Compression Coolers)\nVCC (Vapor Compression Coolers)\nSealed Enclosure Cooling from 1000-5000 Watts\nHigh efficiency and compact vapor compression air conditioners designed for sealed enclosure cooling. This compact product line features plug-and-play capability which ensures fast and effective set-up. A flanged design is incorporated for convenient through-wall (or cabinet door) mounting to efficiently cool your power electronics cabinet.\nHarsh Environment Capability (IP55 seal)\nThe closed-loop cooling system protects equipment from harsh environments, with an adjustable cabinet air temperature setpoint from 20°C to 40°C\nBLDC motor technology enables variable speed control for both fans and compressors\n(Only for DC models)\n1-Year Warranty (see ACT’s\nTerms and Conditions for Online Sales\n) with an expected 10 year lifetime\nOrder through our experienced sales staff or\nBuy Online\nACT VAPOR COMPRESSION AIR CONDITIONER PRODUCTS: VCC Series\n-Scroll right to view table-\nPart Number\nCooling Capacity @35°C /35°C\nVoltage (type)\nWeight:\nbs. (kg)\nHeight\nWidth\nDepth **\nACT-VCC-1000-DC\n1000W\n48V*   (DC)\n46 (21)\n31.22″\n793mm\n15.00″\n381mm\n6.89″\n175mm\nACT-VCC-2000-AC\n2000W\n220-240V (AC)\n71 (32)\n29.33″\n745mm\n17.52″\n445mm\n7.87″\n200mm\nACT-VCC-3000-DC\n3000W\n48V*   (DC)\n104 (47)\n45.30″\n1150mm\n19.09″\n485mm\n8.86″\n225mm\nACT-VCC-5000-AC\n5000W\n220-240V (AC)\n154 (70)\n51.18″\n1300mm\n23.62″\n600mm\n11.81″\n300mm\nNotes:\n*DC units are wired for Negative 48 Volt connection.\n** Insertion Depth: all models extend 1.77”/45mm into the enclosure\nACT – VAPOR COMPRESSION AIR CONDITIONERS (VCC Series): FEATURES AND OPTIONS\nUse our free\nEnclosure Cooler Selection Tool\nto determine the proper model for your cabinet.\nTalk to the Thermal Experts\nFill out the form below to get started\nContact\nTop\nPage Index\nSealed Enclosure Cooling from 1000-5000 Watts\nACT VAPOR COMPRESSION AIR CONDITIONER PRODUCTS: VCC Series\nACT – VAPOR COMPRESSION AIR CONDITIONERS (VCC Series): FEATURES AND OPTIONS\nRelated Products\n5,000 Watt Cabinet Air Conditioner - ACT-VCC-5000-AC\nBUY NOW\n3,000 Watt Cabinet Air Conditioner - ACT-VCC-300\nBUY NOW\n1,000 Watt Cabinet Air Conditioner - ACT-VCC-1000-DC\nBUY NOW\n2,000 Watt Cabinet Air Conditioner - ACT-VCC-2000-AC\nBUY NOW\nRelated Resources\nVCC User Manual\n1000 Watt Spec Sheet\n2000 Watt Spec Sheet\n3000 Watt Spec Sheet\n5000 Watt Spec Sheet\nACT Quality\nContact ACT\nFind your Rep\nAdvanced Cooling Technologies, Inc.\n1046 New Holland Avenue\nLancaster, Pennsylvania  17601, USA\n(717) 295-6061\nContact Our Experts\nlinkedin\nyoutube\ntwitter\nfacebook\nshop products online\nsitemap\nprivacy policy\nterms & conditions\nISO9001 & AS9100 CERTIFIED, ITAR REGISTERED\nCopyright 2024. All rights reserved.", 1),
                ("https://www.a-g.com/wp-content/themes/a-g/css/main.css","@charset \"UTF-8\";\n/*\n *   Font imports\n */\n@import url(\"https://use.typekit.net/nxe4vbn.css\");\n@import url(\"https://use.typekit.net/ays5cfy.css\");\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_bold-webfont.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_bold-webfont.woff\") format(\"font/woff\");\n  font-weight: bold;\n  font-style: normal;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_bold_oblique.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_bold_oblique.woff\") format(\"font/woff\");\n  font-weight: bold;\n  font-style: oblique;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_book_oblique.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_book_oblique.woff\") format(\"font/woff\");\n  font-style: oblique;\n  font-weight: normal;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_book.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_book.woff\") format(\"font/woff\");\n  font-weight: normal;\n  font-style: normal;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_extra-light_oblique.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_extra-light_oblique.woff\") format(\"font/woff\");\n  font-weight: lighter;\n  font-style: oblique;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_extra-light.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_extra-light.woff\") format(\"font/woff\");\n  font-weight: lighter;\n  font-style: normal;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_medium_oblique.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_medium_oblique.woff\") format(\"font/woff\");\n  font-weight: medium;\n  font-style: oblique;\n}\n@font-face {\n  font-family: \"itc-avant-garde-gothic-pro\";\n  src: url(\"../fonts/itc_avant_garde_gothic_lt_medium.ttf\") format(\"truetype\"), url(\"../fonts/itc_avant_garde_gothic_lt_medium.woff\") format(\"font/woff\");\n  font-weight: medium;\n  font-style: normal;\n}\n/*\n *   Typographic declarations\n */\n/*\n  *    (jim halpert stare)", 0),
                ("https://www.1fbusa.com/careers-life","1FBUSA Online Services Careers - Life at 1FBUSA\nCredit Cards\nRespond to Mail Offer\nApply Now\nOur Products\nCredit Card IQ Quiz\nContact Us\nLog In\nBanking\nPersonal Banking\nOverview\nChecking & Savings\nLending\nServices\nContact Us\nLog In\nCorporate Banking\nBuilder Finance\nAbout\nBlog\nScholarship\nLog In\nCareers\nAbout Us\nEmployee Benefits\nLife @ 1FBUSA\nCurrent Opportunities\nLife @ 1FBUSA\n1st Financial Bank USA and its subsidiaries offer a satisfying and challenging results oriented work environment in a small, friendly, business casual, family oriented atmosphere.\nAt 1st Financial Bank USA – We Are:\nF\nirst in providing quality service\nO\npen to innovation and change\nC\nommitted to the needs of our customers\nU\nncompromising in our ethics... Unified as a team... United as an organization\nS\nupportive of the communities we serve\nE\nxperts in information and financial services\nD\nedicated to our Code of Conduct\nWe are\nFOCUSED\n..... We Are\n1st Financial Bank USA\nCareer Areas\nAudit\nCollections\nCommunity Banking\nCompliance\nCustomer Service\nFinance\nHuman Resources\nInformation Systems\nInformation Technology\nLegal\nOperations\nReal Estate\nRisk Management\nABOUT\nBlog\nScholarship\nSUPPORT\nFAQs\nContact Us\nSecurity Center\nLanguage Access Service\nLEGAL\nTerms of Service\nCard Agreements\nPRIVACY\nLINKS\nCareers\nCollegeData\nBuilder Finance\n© 2024 1st Financial Bank USA.\n×\nAll of our communications are in English. We do not provide any language access services. \n        A translation and description of commonly used debt collection terms are available in multiple languages on the \n        New York City Department of Consumer Affairs' website at\nwww.nyc.gov/dca\n.", 0),
                ("http://zuzick.com", "Zuzick Organization | INSURING A BRIGHT FUTURE\nHome\nProtect Your Family\nAbout\nCareers\nContact\nResources\nShop\nApply\nSelect Page\nINSURING A BRIGHT FUTURE\nAPPLY\nZUZICK\nAll our services!\n\nFREE WILL KIT\nOver 65% of Americans today DO NOT have a Will in place! This is alarming because the purpose of a Will is to make sure your assets are properly handled when you die. Our Free Will Kit covers all these issues and MUCH MORE. Best of all, it’s 100% FREE!\n\nCHILD SAFE KIT\nEvery year over 400,000 children are reported missing. This app and physical kit WILL help in this event. It is available at NO COST and endorsed by the National Teachers, School Administrators, and the Police Officers Unions of America.\n\nFAMILY CARE PLAN\nMany financial experts will tell you some unforeseen events and a lack of the right kind of coverage can spell trouble quickly. Call us today, and we’ll analyze your needs and get you protected.\n100% FREE Needs Analysis!\nAll our services!\nTAKE YOUR CAREER TO THE NEXT LEVEL\n“If we expect the best of ourselves, we will also learn to expect the best from everyone around us. If we all work on becoming the best version of ourselves, everyone wins. Everyone and everything around us will grow.”\n– Brian Zuzick, SGA\nJOIN OUR TEAM\nOur mission is to help our clients protect their loved ones from preventable financial distress, give back to the community and create life changing career opportunities\nWe are a full service\nCHECK OUT LIFE INSIDE\n#ZUZICKORGANIZATION\n”WE WANT A MOTIVATING AND INNOVATING CULTURE WHERE EVERYONE WHO JOINS FEELS THEY HAVE OPPORTUNITIES TO SUCCEED AND GROW.”\nTestimonials\nHappy Customers\nAttar Chalif\nPaul Zuzick was the utmost professional. He was helpful and informative. He helped us protect our future family and I cannot recommend this man enough! Zuzick and associates is very lucky to have this man representing the company. I give him 5 stars, I would definity have him over again!\nTomas Robinson\nIt was a wonderful experience end to end obtaining life insurance. My agent Chilynn Acevedo was amazing. She left no page unturned. I learned so many things about certain policies and what they offer. I am grateful for Chilynn's expertise. I highly recommend her and the service she provides. Thank you Chilynn and American Income Life\nmarbela Perez\nI was looking for home mortgage protection and Chilynn was able to walk my husband and I through our options and make the best decision for our current age and financial status. My husband and I had some basic knowledge on life insurance and mortgage protection, but Chilynn made sure we thoroughly understood what it entailed. In addition she made sure that we know she is available as a resource should any questions or life changes come up. Overall, we had a great experience!\nJanelle Liebla\nPaige was extremely knowledgeable and helpful! She took her time to explain everything and went above and beyond answering my many questions. Thank you so much, it is a pleasure working with you!\nKim Key\nAbbyTrifone was very knowledgeable, personable. She takes the time to understand your longterm needs as well as working with your comfort levels of monthly payments. To ensure its reasonable.\r\nPleasant, and a great experience.\nLynnette Chambers\nNathan Mahoney was a great agent to explain to my husband and I the difference in types of insurance. The will process. He was very knowledgeable, he also was very honest. If he did not know the answer to our question. He said he didn’t know but he knew where to turn to get the answer. He has a great team behind him as well who helped answer questions. He was patient with us in the process and made sure we understood the information given. He truly cares about your coverage, with the experience of his personal life. He shared his hardships with us. Which only showed, he is human and not only about business. He is absolutely an 100% an asset...\nAdelino Gomes\nMy meeting with Chandavy was extremely informative. The info she delivered was clear and concise. What is usually uncomfortable subject matter, she was able to create a relaxed environment in which to explain everything to me. At no point did I feel she was trying to force a product on me. I could sense her genuine approach to simply provide myself and my family a coverage that best suit our lifestyle.\nChristine Newray\nAriana D. was amazing to work with. She made the experience very personable and made sure I understood the process the entire way through. Ariana explained every step and made sure I asked any questions as we went on with our meeting. I was referred by a friend as we all went to the same university and I am glad to decided to proceed with Ariana. I never understood much about life insurance but now I can say I know more than when we first began our meeting. Thank you Ariana!\nLOOKING FOR A NEW CAREER?\nLEARN HOW THE AGENCY CAN HELP GROW YOUR FUTURE.\nAPPLY\nCAREER\n\nOffice\n1275 Wampanoag Trail Suite 9 Riverside RI 02915\n\nCall Us\n+1 401-429-1847\n\nEMAIL US\ninfo@zuzick.com\nFacebook\nInstagram\nCopyright © 2024 Zuzick Organization |\nPrivacy\n| Designed By\nALLCAPSMEDIA", 1)
                ]

    instructions = "You are a helpful assistant that decides whether a text is useful or not. \
                    You are going to be helping supply chain managers. \
                    You have to decide if the webpage you are given contains important information for the supply chain manager. \
                    The supply chain manager will ask questions such as:\n \
                    - How many dental practices are there in Zurich?\n\
                    - Which companies in southern Italy produce aluminium auto components?\n\
                    - Which is the cheapest flatbed truck rental in the US?\n\
                    Many pages will not contain this information, but rather contain information about privacy, cookies, etc. Some pages will be the main page of the website, while this information is not the most relevant, please include it. \
                    Very importantly, we want contact information, prices, products, etc. Please just answer with Yes or No.\
                    Some examples of pages and answers are:\n"
    for k,ex,label in examples:
        instructions += f"Page: {k}\n{ex[:max_length]}\nAnswer: " + ("Yes" if label == 1 else "No") + "\n\n"

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=f"Page:  {k}\n{content[:max_length]}\nAnswer: ",
        max_output_tokens=16
    )
    return response.output_text

def process_pages(client, main_page_key: str, pages: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_page = {
            executor.submit(is_interesting, client, key, content): (main_page_key, key)
            for key, content in pages
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_page), total=len(pages), desc=f"Processing {main_page_key}"):
            main_page_key, key = future_to_page[future]
            try:
                is_interesting_result = future.result()
                results.append((main_page_key, key, is_interesting_result))
            except Exception as e:
                print(f"Error processing {key}: {e}")
    return results

if __name__ == "__main__":
    
    load_dotenv()

    client = openai.OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    data_folder = '../data'
    results = {"main_page_key": [], "key": [], "is_interesting": []}

    filename = "interesting_pages.csv"
    if os.path.exists(filename):
        results = pd.read_csv(filename).to_dict(orient="list")

    for file_name in tqdm(os.listdir(data_folder)):
        if file_name.endswith(".json"):
            with open(os.path.join(data_folder, file_name), "r") as file:
                data = json.load(file)
                main_page_key = list(data['text_by_page_url'].keys())[0]
                if main_page_key in results["main_page_key"]:
                    continue
                
                # Prepare pages for parallel processing
                pages = list(data['text_by_page_url'].items())
                
                # Process pages in parallel
                batch_results = process_pages(client, main_page_key, pages)
                
                # Add results to the main results dictionary
                for main_page_key, key, is_interesting_result in batch_results:
                    results["main_page_key"].append(main_page_key)
                    results["key"].append(key)
                    results["is_interesting"].append(is_interesting_result)
                
                # Save after each file is processed
                pd.DataFrame(results).to_csv(filename, index=False)
