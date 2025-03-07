---
layout: post
# layout: single
title:  "Crafting a reward function"
date:   2024-05-09 12:51:28 -0800
categories: jekyll update
---

{% include links/all.md %}

* toc
{:toc}


## Links

 * input parameters description - [https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html)


 * Sample reward functions
   * [https://www.mytechmint.com/advanced-guide-to-aws-deepracer-with-all-tips-and-hacks-to-win-the-race/](https://www.mytechmint.com/advanced-guide-to-aws-deepracer-with-all-tips-and-hacks-to-win-the-race/)
   * [https://github.com/dgnzlz/Capstone_AWS_DeepRacer/blob/master/Reward_Function/reward_function.py](https://github.com/dgnzlz/Capstone_AWS_DeepRacer/blob/master/Reward_Function/reward_function.py)

## Glossary

 State = one image

 CNN feature extractor = The first half of the CNN who extract the parameters from the image.

 CNN classifier = In the case of a discrete state, the second part of the CNN can be thought of as a classifier with as many outputs as there are defined actions. :warning: The classifier takes inputs from other sources than the image, such as the LIDAR, GPS, etc.

 Waypoints = can be used to change the strategy when arriving in a specific section of the track

## Function inputs

The params dictionary object contains the following key-value pairs:

```
{
    "all_wheels_on_track": Boolean,        # flag to indicate if the agent is on the track
    "x": float,                            # agent's x-coordinate in meters
    "y": float,                            # agent's y-coordinate in meters
    "closest_objects": [int, int],         # zero-based indices of the two closest objects to the agent's current position of (x, y).
    "closest_waypoints": [int, int],       # indices of the two nearest waypoints.
    "distance_from_center": float,         # distance from the track center in meters
    "is_crashed": Boolean,                 # Boolean flag to indicate whether the agent has crashed.
    "is_left_of_center": Boolean,          # Flag to indicate if the agent is on the left side to the track center or not.
    "is_offtrack": Boolean,                # Boolean flag to indicate whether the agent has gone off track.
    "is_reversed": Boolean,                # flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    "heading": float,                      # agent's yaw in degrees
    "objects_distance": [float, ],         # list of the objects' distances in meters between 0 and track_length in relation to the starting line.
    "objects_heading": [float, ],          # list of the objects' headings in degrees between -180 and 180.
    "objects_left_of_center": [Boolean, ], # list of Boolean flags indicating whether elements' objects are left of the center (True) or not (False).
    "objects_location": [(float, float),], # list of object locations [(x,y), ...].
    "objects_speed": [float, ],            # list of the objects' speeds in meters per second.
    "progress": float,                     # percentage of track completed
    "speed": float,                        # agent's speed in meters per second (m/s)
    "steering_angle": float,               # agent's steering angle in degrees
    "steps": int,                          # number steps completed
    "track_length": float,                 # track length in meters.
    "track_width": float,                  # width of the track in meters
    "waypoints": [(float, float), ]        # list of (x,y) as milestones along the track center

}
```

```
all_wheels_on_track  True
x                    0.011851113112954031
y                    -3.3566120265028037
closest_waypoints    [1, 2]
distance_from_center 1.8560450670001976e-09
is_crashed           False
is_left_of_center    False
is_offtrack          False
is_reverse           False
heading              -28.76732457199736
progress             0.2784336592054098
speed                1.5
steeering_angle      -15.0
steps                2.0
track_length         50.28127719885948
track_width          1.0667999027394186
waypoints            [(-0.11087026589960641, -3.289236488410292), (0.003085300326347351, -3.351799488067627), ...
 ```

 ```
#WAYPOINT FOR ACE SPEEDWAY TRACK
[(-0.11087026589960641, -3.289236488410292), (0.003085300326347351, -3.351799488067627), (0.11704087816426867, -3.414362466574292), (0.2671550349332392, -3.4967769384384155), (0.5312242358922958, -3.6417551040649414), (0.7952920496463776, -3.786737084388733), (1.059357911348343, -3.931722044944763), (1.3234224915504456, -4.076709985733032), (1.5874925255775452, -4.221686840057373), (1.851571500301361, -4.366644978523254), (2.115659534931186, -4.511584043502809), (2.3797525167465166, -4.656513452529905), (2.6438030004501343, -4.801531553268433), (2.907799482345581, -4.946660995483398), (3.171742558479309, -5.091902017593384), (3.4357104301452637, -5.2370924949646), (3.6999679803848267, -5.381678581237793), (3.964537501335144, -5.525612831115723), (4.229419589042664, -5.668896436691284), (4.493736505508423, -5.813358306884766), (4.756242513656613, -5.961596488952635), (5.01692867279053, -6.113630056381227), (5.27595853805542, -6.269117593765259), (5.541237831115723, -6.411571979522705), (5.816755056381226, -6.532676935195923), (6.101351976394653, -6.629037380218506), (6.396347284317015, -6.689000844955444), (6.6969330310821515, -6.691721439361572), (6.9932332038879395, -6.643845081329346), (7.277515888214111, -6.546059846878052), (7.534146308898926, -6.389546155929565), (7.776738405227661, -6.2113964557647705), (7.987441301345825, -5.996051549911499), (8.177835464477539, -5.763434410095215), (8.352219581604004, -5.5181074142456055), (8.50377082824707, -5.257759094238281), (8.639700412750244, -4.98923397064209), (8.765072345733643, -4.715397596359253), (8.871030330657959, -4.433393955230713), (8.966260433197021, -4.147768974304199), (9.051724433898926, -3.8589255809783936), (9.11946725845337, -3.5653860569000244), (9.175206661224365, -3.2694865465164185), (9.21681547164917, -2.9713035821914673), (9.239078521728516, -2.6710569858551025), (9.231091499328613, -2.3702094554901123), (9.18811559677124, -2.0723780393600464), (9.107061386108398, -1.7828075289726257), (8.982296466827393, -1.5093684792518616), (8.810195446014404, -1.2630034387111664), (8.590136051177979, -1.0581815540790558), (8.339211463928223, -0.892494410276413), (8.066346168518066, -0.7656736373901367), (7.779706001281738, -0.6735609769821167), (7.486222982406616, -0.6062817797064781), (7.18891453742981, -0.5583668043836951), (6.889522075653076, -0.5256308731622994), (6.589047908782959, -0.5046488028019667), (6.288018465042114, -0.49526200257241726), (5.986917495727539, -0.5001590140163898), (5.686349630355835, -0.5194105338305235), (5.386559963226318, -0.5488441241905093), (5.087057113647461, -0.5812348667532206), (4.786983013153076, -0.6075533777475357), (4.485960006713867, -0.6167096272110939), (4.1851654052734375, -0.6039147675037384), (3.8864705562591553, -0.5665732230991125), (3.591686964035034, -0.5055527011863887), (3.301775574684143, -0.4242045059800148), (3.0170695781707764, -0.32573433965444565), (2.7393590211868286, -0.20908604562282562), (2.4659245014190674, -0.08296085894107819), (2.196008026599884, 0.050819799304008484), (1.9324599504470825, 0.19667485356330872), (1.6699939966201782, 0.34449559077620506), (1.415526032447815, 0.5057430770248175), (1.1642137169837952, 0.6715977564454079), (0.921069860458374, 0.8494530618190765), (0.6870073676109314, 1.0388080477714539), (0.46010078117251396, 1.23677659034729), (0.2466164380311966, 1.4493204951286316), (0.04697015881538391, 1.6746694445610046), (-0.13940425217151642, 1.9110100269317627), (-0.3099997937679291, 2.159074544906616), (-0.4627888516988605, 2.418657064437866), (-0.5940006338059902, 2.6898410320281982), (-0.7041242122650146, 2.9701755046844482), (-0.7941821217536926, 3.257549524307251), (-0.8645032346248627, 3.5503724813461304), (-0.9162373691797256, 3.8470590114593506), (-0.9513023942708969, 4.146209001541138), (-0.9694564789533615, 4.446866989135742), (-0.9822502732276917, 4.747759819030762), (-1.0381899178028107, 5.042850017547607), (-1.1555785536766052, 5.319516897201538), (-1.3141512870788574, 5.575245141983032), (-1.5018789768218994, 5.810555458068848), (-1.715216040611267, 6.023006439208984), (-1.9470694661140442, 6.21501350402832), (-2.1957759857177734, 6.384726047515869), (-2.46309757232666, 6.523125886917114), (-2.7448114156723022, 6.6288065910339355), (-3.0397114753723145, 6.68622899055481), (-3.3405975103378287, 6.693716049194336), (-3.6418234109878544, 6.68273401260376), (-3.9429715871810913, 6.675971031188965), (-4.244132995605469, 6.6685004234313965), (-4.545292139053345, 6.661149024963379), (-4.8464515209198, 6.6537768840789795), (-5.147611141204834, 6.64640736579895), (-5.4487714767456055, 6.6390464305877686), (-5.749925851821899, 6.631633996963501), (-6.051111698150635, 6.62451958656311), (-6.352118492126465, 6.6156604290008545), (-6.654168367385864, 6.617000102996826), (-6.951627016067505, 6.573275804519653), (-7.245448112487793, 6.507750511169434), (-7.531482934951782, 6.414634466171265), (-7.7983012199401855, 6.2765583992004395), (-8.012423038482666, 6.065863370895386), (-8.191925525665283, 5.824285507202148), (-8.350342512130737, 5.568169355392456), (-8.492345333099365, 5.302488565444946), (-8.619697570800781, 5.0295326709747314), (-8.736942768096924, 4.752150058746338), (-8.845244884490967, 4.471105575561523), (-8.94164514541626, 4.185697078704834), (-9.02610731124878, 3.8966175317764282), (-9.100931167602539, 3.604933023452759), (-9.16403579711914, 3.3103615045547485), (-9.207085132598877, 3.0122464895248413), (-9.232807636260986, 2.7123879194259644), (-9.235105991363525, 2.4116299152374268), (-9.203819274902344, 2.112697958946228), (-9.121238708496092, 1.824334025382994), (-8.964235305786133, 1.5708979964256287), (-8.735591888427733, 1.3771349787712088), (-8.485531330108643, 1.209145426750183), (-8.228861093521116, 1.0515300929546347), (-7.96850061416626, 0.9001266658306122), (-7.703809499740599, 0.7562850713729851), (-7.434115409851074, 0.6220604628324509), (-7.16211462020874, 0.4926215880550444), (-6.888531446456909, 0.3665221035480499), (-6.614516019821165, 0.24135971069335863), (-6.341501951217654, 0.1140217036008846), (-6.068720102310181, -0.013770103454589844), (-5.800762891769409, -0.151077538728714), (-5.538833856582642, -0.30027854442596436), (-5.277379512786865, -0.45041576866060495), (-5.0148091316223145, -0.5983506664633751), (-4.751121520996094, -0.7440836876630783), (-4.48656153678894, -0.8880929797887802), (-4.222092628479004, -1.0322833061218262), (-3.957815408706665, -1.176851212978363), (-3.6937299966812134, -1.321797102689743), (-3.4297515153884888, -1.4669555723667145), (-3.1657429933547974, -1.6120535731315613), (-2.901700973510742, -1.7570860385894775), (-2.6376270055770874, -1.9020545482635498), (-2.3735435009002686, -2.0470050573349), (-2.109465479850769, -2.191966474056244), (-1.8453929424285889, -2.336938500404358), (-1.5813264846801758, -2.481921076774597), (-1.3172594904899597, -2.626904606819153), (-1.0531918704509735, -2.771886467933655), (-0.7891235053539276, -2.9168660640716553), (-0.525053858757019, -3.061843991279602), (-0.26098431809805334, -3.2068220376968384), (-0.11087026589960641, -3.289236488410292)]
```

 More at:
  * input parameters description - [https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html](https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html)


## Log analysis

```
                  x      y       heading
SIM_TRACE_LOG:0,1,0.0120,-3.3567,-28.7719,-15.00,1.50,0,0.0000,False,True,0.2787,1,50.28,20.315,prepare,0.00
SIM_TRACE_LOG:0,2,0.0119,-3.3566,-28.7673,-15.00,1.50,0,1.0000,False,True,0.2784,1,50.28,20.382,in_progress,0.00
```

## Code snippet

### Print Parameters

 Everything printed in the reward function is printer in the robomaker log file.

 ```
 def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''

    # Read input parameters
    all_wheels_on_track = params['all_wheels_on_track']
    x = params['x']
    y = params['y']
    closest_objects = params['closest_objects']
    closest_waypoints = params['closest_waypoints']
    distance_from_center = params['distance_from_center']
    is_crashed = params['is_crashed']
    is_left_of_center = params['is_left_of_center']
    is_offtrack = params['is_offtrack']
    is_reversed = params['is_reversed']
    heading = params['heading']
    objects_distance = params['objects_distance']
    objects_heading = params['objects_heading']
    objects_left_of_center = params['objects_left_of_center']
    objects_location = params['objects_location']
    objects_speed = params['objects_speed']
    progress = params['progress']
    speed = params['speed']
    steering_angle = params['steering_angle']
    steps = params['steps']
    track_length = params['track_length']
    track_width = params['track_width']
    waypoints = params['waypoints']

    print("awot", all_wheels_on_track, " x", x, " y", y, " cw", closest_waypoints, " dfc", distance_from_center, " is", is_crashed, " iloc", is_left_of_center, " io", is_offtrack, " ir",  is_reversed, " h", heading, " p", progress, " s", speed, " sa", steering_angle, " s", steps, " tl", track_length, " tw", track_width, " w", waypoints)

    ...
 ```

## Sample reward functions

 Source - [https://www.mytechmint.com/advanced-guide-to-aws-deepracer-with-all-tips-and-hacks-to-win-the-race/](https://www.mytechmint.com/advanced-guide-to-aws-deepracer-with-all-tips-and-hacks-to-win-the-race/)
