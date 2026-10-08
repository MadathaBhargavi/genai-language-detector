"""
Curated training and benchmark dataset for the GenAI Language Detector.
Contains paired Human-written (label=0) and AI-generated (label=1) texts
across diverse domains: essays, emails, tech overviews, creative prose, and discourse.
"""

TRAINING_DATA = [
    # ==========================================
    # HUMAN SAMPLES (label = 0)
    # ==========================================
    {
        "text": """I honestly didn't expect much when I first set up the backyard smoker last Saturday. 
My brother kept saying the temperature would fluctuate wildly because of the sudden chilly wind coming off the ridge, and for the first two hours he was totally right. 
I had to stand out there in an old flannel shirt, poking at the damp oak chunks with a pair of rusty grill tongs, wondering why I didn't just order barbecue takeout like a normal person. 
Around hour four, though, the thin blue smoke settled down. The crust on the brisket turned into that dark mahogany mahogany bark everyone obsesses over. 
When we finally carved into it on our picnic table, nobody said a word for ten straight minutes—we were just eating like wolves. Best Saturday afternoon all autumn.""",
        "label": 0
    },
    {
        "text": """Reading through the latest quarter's performance metrics left me with more questions than answers. 
The leadership deck claims our user acquisition costs dropped by 18%, but if you look at the raw cohort retention table on page 42, churn among 30-day cohorts actually spiked. 
What gives? I poked around our internal analytics dashboard and noticed we stopped counting trial cancellations that happened within the first 48 hours as active churn. 
That's not genuine growth; that's just moving the goalposts so the executive summary looks shiny for Thursday's board call. 
I'm drafting an email to Dave tonight to ask for the unadjusted numbers before we commit to the Q3 headcount plan.""",
        "label": 0
    },
    {
        "text": """The morning train from New Haven was delayed again, surprise surprise. 
A signal malfunction right outside Stamford left hundreds of us stranded on the platform, holding lukewarm coffees and glaring at the flickering display board. 
A guy next to me was furiously tapping his mechanical watch against his briefcase, muttering under his breath about a 9:00 AM pitch in Midtown. 
I gave up, found an empty bench near the newspaper kiosk, and finished two chapters of my paperback. 
Sometimes the only sane way to deal with commuter chaos is just accepting that the schedule isn't yours to control today.""",
        "label": 0
    },
    {
        "text": """When we switched our legacy billing pipeline over to asynchronous worker queues back in 2021, everyone braced for disaster. 
Our senior architect spent three nights sleeping under his desk, convinced that race conditions would double-charge half our enterprise tier. 
It turned out the real bottleneck wasn't the queue at all—it was an innocent-looking index lock on our primary Postgres transactions table that had been lurking there since the original seed script in 2017. 
Once we dropped that index and replayed the dead-letter events, the whole backlog cleared out in twenty minutes flat. We ended up ordering celebratory cold beer at 2 AM.""",
        "label": 0
    },
    {
        "text": """I've been tinkering with water coloring lately, and frankly, my trees look like spilled guacamole. 
There's something deeply humbling about watching pigment bleed uncontrollably into wet cold-press paper when you thought you had full command of the brush. 
My art teacher always says to 'let the water do the thinking,' but my brain hates relinquishing control. 
Still, even when a painting turns into an unrecognizable brown smudge, the ritual of mixing cobalt blue with yellow ochre on the porcelain palette feels therapeutic after staring at monitor screens for nine hours.""",
        "label": 0
    },
    {
        "text": """Listen, I know everyone is hyped about the new framework release, but let's be realistic for five seconds. 
We still have three microservices running on a deprecated runtime version that nobody has dared touch since Elena left the company last November. 
If we introduce another paradigm shift now, we'll spend our entire summer resolving dependency graph conflicts and writing wrapper shims instead of shipping actual user-facing features. 
Let's pay down our existing technical debt first before chasing the shiny new toy.""",
        "label": 0
    },
    {
        "text": """My grandmother's kitchen always smelled like toasted cardamom and sizzling mustard seeds. 
She never owned a measuring spoon; everything was measured against the curvature of her palm or the sound of the splutter in the heavy iron kadai. 
Whenever I tried replicating her lentil stew in my cramped apartment kitchen, something was always off—either too salty, or missing that distinct smoky aroma she coaxed out of dried red chilies. 
Last Thanksgiving, I finally sat beside her on a wooden stool and wrote down her tiny subconscious micro-adjustments in a dog-eared notebook.""",
        "label": 0
    },
    {
        "text": """We spent the entire afternoon troubleshooting why our Bluetooth peripheral wouldn't pair with the updated test bench. 
Turned out some intern had accidentally swapped the firmware hex files on the shared server, so we were flashing an obsolete BLE 4.0 stack onto a modern chip. 
Five engineers staring at oscilloscope traces for four hours, all because of an ambiguous filename like 'firmware_final_v2_new.bin'. You really have to laugh, or else you'll pull your hair out.""",
        "label": 0
    },
    {
        "text": """The bookstore downtown has this curious policy where you can trade any three used paperbacks for a cup of drip coffee and a seat in their basement reading nook. 
It's damp down there, and the floorboards creak every time someone wanders into the biography section, but the quiet is sacred. 
I spent three hours there last Sunday reading an obscure 1970s sci-fi novel about deep-sea mining colonies. 
No notifications, no buzzing watch, just the smell of dusty yellowed paper and rain tapping on the street-level grates above.""",
        "label": 0
    },
    {
        "text": """I don't buy the argument that remote work destroys team camaraderie. 
If your company culture relied exclusively on bumping into each other by the water cooler or forced awkward happy hours with lukewarm pizza, you didn't have a culture—you had a shared physical prison. 
Our remote squad shares ridiculous memes on Slack, hops on spontaneous 10-minute pairing calls to unblock tough bugs, and respects each other's boundaries when deep focus is required. That's real trust.""",
        "label": 0
    },
    {
        "text": """My dog has this bizarre habit of burying her chew bones in the laundry basket. 
You reach in to grab a clean towel after a shower, and boom, you get a mouthful of dried rawhide. 
I've tried moving the basket to the closet, but she figured out how to nudge the folding door open with her snout. 
At this point, I've just accepted that my socks will permanently smell slightly of smoked beef.""",
        "label": 0
    },
    {
        "text": """Writing documentation is easily the most thankless task in software engineering until you come back to your own repository six months later. 
Yesterday I found a script I wrote in a hurry last December. Zero comments, variables named 'tmp_data_2', and an undocumented regular expression that looked like an alien distress signal. 
I spent forty-five minutes reverse-engineering my own logic, cursing past-me the entire time. Document your code, folks. If not for your teammates, do it out of pity for your future self.""",
        "label": 0
    },

    # ==========================================
    # AI SAMPLES (label = 1)
    # ==========================================
    {
        "text": """Artificial intelligence has fundamentally transformed the modern technological landscape, reshaping how industries operate and innovate. 
In today's fast-paced digital era, machine learning algorithms play a pivotal role in optimizing operational workflows, enhancing predictive accuracy, and fostering data-driven decision-making. 
Furthermore, it is important to note that the seamless integration of neural networks across sectors such as healthcare and finance underscores the vital importance of robust governance frameworks. 
As organizations continue to navigate this dynamic paradigm, striking a harmonious balance between rapid innovation and ethical considerations remains paramount. 
In conclusion, the future of artificial intelligence holds immense promise, offering a rich tapestry of opportunities for sustainable societal advancement.""",
        "label": 1
    },
    {
        "text": """To effectively cultivate professional productivity in a hybrid work environment, organizations must establish clear communication channels and transparent expectations. 
First and foremost, leveraging collaborative digital platforms enables distributed teams to maintain alignment across ongoing strategic initiatives. 
Moreover, fostering an inclusive company culture empowers individual contributors to take ownership of their deliverables while mitigating the risks of workplace burnout. 
It is worth noting that regular feedback mechanisms and structured milestone reviews provide crucial guidance for sustained organizational growth. 
Ultimately, by embracing adaptability and nurturing empathy, modern enterprises can seamlessly navigate the complexities of contemporary workforce dynamics.""",
        "label": 1
    },
    {
        "text": """Sustainable urban development represents a critical cornerstone in combating global climate change and ensuring long-term ecological resilience. 
By integrating renewable energy infrastructures, intelligent public transit systems, and energy-efficient building standards, municipal leaders can significantly reduce metropolitan carbon footprints. 
Furthermore, the strategic preservation of green spaces plays a vital role in enhancing air quality and promoting biodiversity within urban ecosystems. 
Crucially, community engagement and participatory planning methodologies ensure that developmental policies remain equitable and responsive to citizen needs. 
In summary, a holistic approach to urban planning serves as a testament to humanity's collective capacity for harmonious coexistence with nature.""",
        "label": 1
    },
    {
        "text": """Delving into the multifaceted realm of cybersecurity reveals an intricate tapestry of emerging threats and defensive countermeasures. 
In an increasingly interconnected digital ecosystem, safeguarding sensitive data assets requires a comprehensive, defense-in-depth posture. 
Organizations must proactively implement robust encryption protocols, multi-factor authentication mechanisms, and continuous threat monitoring frameworks. 
Additionally, fostering a proactive culture of security awareness among employees serves as a pivotal barrier against sophisticated phishing vectors and social engineering vulnerabilities. 
Consequently, maintaining operational vigilance and adhering to industry compliance benchmarks is essential for bolstering institutional resilience.""",
        "label": 1
    },
    {
        "text": """The transition toward renewable energy sources marks a profound paradigm shift in global industrial infrastructure. 
Solar photovoltaic arrays, wind turbine installations, and advanced hydroelectric systems offer viable alternatives to traditional fossil fuels, substantially mitigating greenhouse gas emissions. 
Moreover, continuous advancements in grid-scale battery storage technologies address intermittent generation challenges, ensuring steady baseload reliability. 
It is important to emphasize that widespread policy incentives and public-private partnerships accelerate capital deployment into sustainable energy ventures. 
To conclude, the global adoption of green energy solutions underscores a pivotal commitment toward achieving carbon neutrality and fostering intergenerational environmental stewardship.""",
        "label": 1
    },
    {
        "text": """Effective time management is a cornerstone of both academic achievement and long-term career progression. 
By establishing structured daily routines and prioritizing high-impact objectives through proven methodologies, individuals can significantly optimize their cognitive output. 
Furthermore, minimizing digital distractions and cultivating uninterrupted focus blocks enhances overall creative problem-solving capabilities. 
It is worth noting that adequate rest and mindful self-care routines are equally vital in preventing mental exhaustion and sustaining peak performance. 
In summary, cultivating disciplined time allocation habits empowers professionals to achieve their aspirations with purpose and efficiency.""",
        "label": 1
    },
    {
        "text": """The culinary arts serve as a powerful testament to the vibrant interplay between cultural tradition and creative sensory experimentation. 
Across various geographic regions, culinary techniques reflect deep-seated communal values, local agrarian practices, and historical heritage. 
Furthermore, contemporary gastronomy frequently embraces cross-cultural fusion, harmonizing traditional flavor profiles with innovative modern culinary methods. 
This delicate balance underscores the enduring significance of food as a universal language that fosters mutual understanding and human connection. 
Ultimately, the exploration of diverse culinary expressions offers a rich, sensory journey that enriches our shared global consciousness.""",
        "label": 1
    },
    {
        "text": """Cloud computing has emerged as a transformative technological pillar, empowering modern organizations with unprecedented scalability and operational agility. 
By migrating computational workloads to distributed cloud infrastructures, businesses can dramatically reduce physical hardware expenditures while optimizing service availability. 
Additionally, serverless architectures and containerized orchestration platforms facilitate continuous deployment pipelines and rapid feature iteration. 
Nevertheless, implementing rigorous identity access management policies and continuous telemetry monitoring remains vital for safeguarding sensitive multi-tenant environments. 
In conclusion, embracing modern cloud architectures provides organizations with the foundational flexibility necessary to thrive in an ever-evolving digital landscape.""",
        "label": 1
    },
    {
        "text": """The exploration of outer space represents humanity's perpetual quest to unravel the fundamental mysteries of the cosmos. 
Robotic exploration missions, orbital space observatories, and interplanetary probes have profoundly expanded our understanding of planetary geology and stellar evolution. 
Furthermore, international collaboration in astronomical research highlights our shared aspiration to transcend terrestrial boundaries. 
As private aerospace enterprises and governmental agencies embark on ambitious missions to the Moon and Mars, ethical frameworks and planetary protection guidelines become paramount. 
In summary, space exploration continues to inspire technological innovation while expanding the horizons of human knowledge.""",
        "label": 1
    },
    {
        "text": """Critical thinking constitutes an indispensable intellectual skill set in an era inundated with vast amounts of digital information. 
The ability to systematically dissect arguments, evaluate empirical evidence, and identify cognitive biases enables individuals to make sound, reasoned determinations. 
Moreover, cultivating an intellectually curious mindset encourages lifelong learning and adaptive problem-solving across complex domains. 
It is important to recognize that reflective discourse and open-minded inquiry foster civil debate and intellectual resilience. 
Ultimately, honing analytical reasoning capabilities serves as a crucial foundation for informed citizenship and ethical leadership in contemporary society.""",
        "label": 1
    },
    {
        "text": """The integration of telemedicine into contemporary healthcare systems has fundamentally revolutionized patient care accessibility and clinical workflows. 
By offering secure virtual consultations, remote patient monitoring tools, and asynchronous health management portals, providers can effectively bridge geographic healthcare disparities. 
Furthermore, digital health platforms streamline administrative overhead and optimize appointment scheduling efficiency. 
However, addressing digital literacy barriers and ensuring equitable broadband connectivity remain crucial prerequisites for achieving universal telehealth adoption. 
In conclusion, the ongoing digital transformation of healthcare holds immense potential to enhance patient outcomes and democratize quality clinical support.""",
        "label": 1
    },
    {
        "text": """The preservation of historical literature plays a vital role in safeguarding human cultural heritage and collective memory. 
Through meticulous archival conservation and state-of-the-art digital preservation techniques, rare manuscripts and seminal literary works are protected for future generations. 
Moreover, scholarly analysis of ancient texts offers invaluable insights into the linguistic nuances, philosophical tenets, and societal norms of bygone eras. 
It is worth noting that making these historical treasures accessible online fosters global educational equity and cultural exchange. 
To summarize, preserving classic literature ensures that humanity's intellectual legacy continues to inspire and enlighten scholars worldwide.""",
        "label": 1
    }
]
